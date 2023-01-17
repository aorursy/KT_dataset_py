!pip install transformers > /dev/null

!pip install pandarallel > /dev/null
import numpy as np

import pandas as pd



import os

os.environ['XLA_USE_BF16'] = "1"



import torch

import torch.nn as nn

from torch.utils.data import Dataset,DataLoader

from torch.utils.data.sampler import SequentialSampler



import time

import random

from datetime import datetime

from tqdm import tqdm

tqdm.pandas()



from transformers import XLMRobertaModel, XLMRobertaTokenizer, XLMRobertaConfig

from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule



import re



# !pip install nltk > /dev/null

import nltk

nltk.download('punkt')



from nltk import sent_tokenize



from pandarallel import pandarallel



pandarallel.initialize(nb_workers=2, progress_bar=True)
!wget -O en-ru.txt.zip http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/moses/en-ru.txt.zip

!unzip en-ru.txt.zip

!rm en-ru.txt.zip

!rm OpenSubtitles.en-ru.ids
!head -10 OpenSubtitles.en-ru.en
!head -10 OpenSubtitles.en-ru.ru
ru_file = open('./OpenSubtitles.en-ru.ru', 'r')

en_file = open('./OpenSubtitles.en-ru.en', 'r')



dataset = {'en': [],'ru': []}



total_lines = 10000

for i in tqdm(range(total_lines), total=total_lines):

    ru_text = ru_file.readline()

    en_text = en_file.readline()



    if not en_text and not ru_text:

        # one of file is finished

        break



    ru_text = ru_text.strip()

    en_text = en_text.strip()

    if not en_text or not ru_text:

        continue



    dataset['ru'].append(ru_text)

    dataset['en'].append(en_text)

    

ru_file, en_file = None, None



del ru_file

del en_file



df = pd.DataFrame(dataset)

df.head()
!pip install spacy > /dev/null

!pip install spacy_cld > /dev/null

!python -m spacy download xx_ent_wiki_sm > /dev/null



import spacy

from spacy_cld import LanguageDetector

import xx_ent_wiki_sm



nlp = xx_ent_wiki_sm.load()

language_detector = LanguageDetector()

nlp.add_pipe(language_detector)
def get_lang_score(text):

    try:

        doc = nlp(str(text))

        language_scores = doc._.language_scores

        return language_scores.get('ru', 0)

    except Exception:

        return 0



text = df.iloc[0]['ru']

print(f'[{get_lang_score(text)}]', text)
df['lang_score'] = df['ru'].parallel_apply(get_lang_score)

df['lang_score'].hist(bins=100)
df = df[df['lang_score'] > 0.8]
MAX_LENGTH = 224

BACKBONE_PATH = '../input/multitpu-inference'

CHECKPOINT_PATH = '../input/multitpu-inference/checkpoint-xlm-roberta.bin'



LANGS = {

    'en': 'english',

    'it': 'italian', 

    'fr': 'french', 

    'es': 'spanish',

    'tr': 'turkish', 

    'ru': 'russian',

    'pt': 'portuguese'

}



def get_sentences(text, lang='en'):

    return sent_tokenize(text, LANGS.get(lang, 'english'))



def exclude_duplicate_sentences(text, lang='en'):

    sentences = []

    for sentence in get_sentences(text, lang):

        sentence = sentence.strip()

        if sentence not in sentences:

            sentences.append(sentence)

    return ' '.join(sentences)



def clean_text(text, lang='en'):

    text = str(text)

    text = re.sub(r'[0-9"]', '', text)

    text = re.sub(r'#[\S]+\b', '', text)

    text = re.sub(r'@[\S]+\b', '', text)

    text = re.sub(r'https?\S+', '', text)

    text = re.sub(r'\s+', ' ', text)

    text = exclude_duplicate_sentences(text, lang)

    return text.strip()
class DatasetRetriever(Dataset):



    def __init__(self, df):

        self.comment_texts = df['en'].values

        self.ids = df.index.values

        self.tokenizer = XLMRobertaTokenizer.from_pretrained(BACKBONE_PATH)



    def get_tokens(self, text):

        encoded = self.tokenizer.encode_plus(

            text, 

            add_special_tokens=True, 

            max_length=MAX_LENGTH, 

            pad_to_max_length=True

        )

        return encoded['input_ids'], encoded['attention_mask']



    def __len__(self):

        return self.ids.shape[0]



    def __getitem__(self, idx):

        text = self.comment_texts[idx]



        tokens, attention_mask = self.get_tokens(text)

        tokens, attention_mask = torch.tensor(tokens), torch.tensor(attention_mask)



        return self.ids[idx], tokens, attention_mask
%%time



df['en'] = df.parallel_apply(lambda x: clean_text(x['en'], 'en'), axis=1)

df = df.drop_duplicates(subset='en')

df = df.drop_duplicates(subset='ru')
class ToxicSimpleNNModel(nn.Module):



    def __init__(self, backbone):

        super(ToxicSimpleNNModel, self).__init__()

        self.backbone = backbone

        self.dropout = nn.Dropout(0.3)

        self.linear = nn.Linear(

            in_features=self.backbone.pooler.dense.out_features*2,

            out_features=2,

        )



    def forward(self, input_ids, attention_masks):

        bs, seq_length = input_ids.shape

        seq_x, _ = self.backbone(input_ids=input_ids, attention_mask=attention_masks)

        apool = torch.mean(seq_x, 1)

        mpool, _ = torch.max(seq_x, 1)

        x = torch.cat((apool, mpool), 1)

        x = self.dropout(x)

        return self.linear(x)





backbone = XLMRobertaModel(XLMRobertaConfig.from_pretrained(BACKBONE_PATH))
class GPUPredictor:

    

    def __init__(self, model):

        self.device = torch.device('cuda:0')

        self.model = model.to(self.device)



    def run_inference(self, loader):

        self.model.eval()

        result = {'id': [], 'toxic': []}

        for step, (ids, inputs, attention_masks) in tqdm(enumerate(loader), total=len(loader)):

            with torch.no_grad():

                inputs = inputs.to(self.device) 

                attention_masks = attention_masks.to(self.device)

                outputs = self.model(inputs, attention_masks)

                toxics = nn.functional.softmax(outputs, dim=1).data.cpu().numpy()[:,1]



            result['id'].extend(ids.numpy())

            result['toxic'].extend(toxics)



        return pd.DataFrame(result)
net = ToxicSimpleNNModel(backbone=backbone)

checkpoint = torch.load(CHECKPOINT_PATH)

net.load_state_dict(checkpoint);



checkpoint = None

del checkpoint
dataset = DatasetRetriever(df)

loader = torch.utils.data.DataLoader(

    dataset,

    batch_size=16,

    sampler=SequentialSampler(dataset),

    pin_memory=False,

    drop_last=False,

    num_workers=2

)



predictor = GPUPredictor(net)
predictions = predictor.run_inference(loader)

predictions = predictions.set_index('id')
df.loc[predictions.index, 'toxic'] = predictions['toxic']

df = df[(df['toxic'] > 0.9) | (df['toxic'] < 0.2)]

df.loc[:, 'toxic'] = df['toxic'].round().astype(int)

df['toxic'].value_counts()
for _, row in df[df['toxic'] == 1].head(10).iterrows():

    print('-'*10)

    print('[EN]')

    print(row['en'])

    print('[RU]')

    print(row['ru'])