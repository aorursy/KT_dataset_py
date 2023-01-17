!pip install -q transformers

!pip install num2words
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pathlib import Path



import os



import torch

import torch.optim as optim



import random



# fastai

from fastai import *

from fastai.text import *

from fastai.callbacks import *



# transformers

from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig



from transformers import BertForSequenceClassification, BertTokenizer, BertConfig

from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig

from transformers import XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig

from transformers import XLMForSequenceClassification, XLMTokenizer, XLMConfig

from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig
# Packages for cleaning texts

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()



import unidecode

import spacy

from num2words import num2words



# Stopwords

from spacy.lang.en.stop_words import STOP_WORDS

STOP_WORDS = list(STOP_WORDS)

exclude = ['no', 'nor', 'not']

for w in exclude:

    try:

        STOP_WORDS.remove(exclude)

    except:

        pass
contractions = { 

"ain't": "am not / are not / is not / has not / have not",

"aren't": "are not / am not",

"can't": "cannot",

"can't've": "cannot have",

"'cause": "because",

"could've": "could have",

"couldn't": "could not",

"couldn't've": "could not have",

"didn't": "did not",

"doesn't": "does not",

"don't": "do not",

"hadn't": "had not",

"hadn't've": "had not have",

"hasn't": "has not",

"haven't": "have not",

"he'd": "he had / he would",

"he'd've": "he would have",

"he'll": "he shall / he will",

"he'll've": "he shall have / he will have",

"he's": "he has / he is",

"how'd": "how did",

"how'd'y": "how do you",

"how'll": "how will",

"how's": "how has / how is / how does",

"I'd": "I had / I would",

"I'd've": "I would have",

"I'll": "I shall / I will",

"I'll've": "I shall have / I will have",

"I'm": "I am",

"I've": "I have",

"isn't": "is not",

"it'd": "it had / it would",

"it'd've": "it would have",

"it'll": "it shall / it will",

"it'll've": "it shall have / it will have",

"it's": "it has / it is",

"let's": "let us",

"ma'am": "madam",

"mayn't": "may not",

"might've": "might have",

"mightn't": "might not",

"mightn't've": "might not have",

"must've": "must have",

"mustn't": "must not",

"mustn't've": "must not have",

"needn't": "need not",

"needn't've": "need not have",

"o'clock": "of the clock",

"oughtn't": "ought not",

"oughtn't've": "ought not have",

"shan't": "shall not",

"sha'n't": "shall not",

"shan't've": "shall not have",

"she'd": "she had / she would",

"she'd've": "she would have",

"she'll": "she shall / she will",

"she'll've": "she shall have / she will have",

"she's": "she has / she is",

"should've": "should have",

"shouldn't": "should not",

"shouldn't've": "should not have",

"so've": "so have",

"so's": "so as / so is",

"that'd": "that would / that had",

"that'd've": "that would have",

"that's": "that has / that is",

"there'd": "there had / there would",

"there'd've": "there would have",

"there's": "there has / there is",

"they'd": "they had / they would",

"they'd've": "they would have",

"they'll": "they shall / they will",

"they'll've": "they shall have / they will have",

"they're": "they are",

"they've": "they have",

"to've": "to have",

"wasn't": "was not",

"we'd": "we had / we would",

"we'd've": "we would have",

"we'll": "we will",

"we'll've": "we will have",

"we're": "we are",

"we've": "we have",

"weren't": "were not",

"what'll": "what shall / what will",

"what'll've": "what shall have / what will have",

"what're": "what are",

"what's": "what has / what is",

"what've": "what have",

"when's": "when has / when is",

"when've": "when have",

"where'd": "where did",

"where's": "where has / where is",

"where've": "where have",

"who'll": "who shall / who will",

"who'll've": "who shall have / who will have",

"who's": "who has / who is",

"who've": "who have",

"why's": "why has / why is",

"why've": "why have",

"will've": "will have",

"won't": "will not",

"won't've": "will not have",

"would've": "would have",

"wouldn't": "would not",

"wouldn't've": "would not have",

"y'all": "you all",

"y'all'd": "you all would",

"y'all'd've": "you all would have",

"y'all're": "you all are",

"y'all've": "you all have",

"you'd": "you had / you would",

"you'd've": "you would have",

"you'll": "you shall / you will",

"you'll've": "you shall have / you will have",

"you're": "you are",

"you've": "you have"

}
import fastai

import transformers

print('fastai version :', fastai.__version__)

print('transformers version :', transformers.__version__)
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
base_dir =  '/kaggle/input/tridata'

train = pd.read_csv(os.path.join(base_dir, 'train.csv'))

test = pd.read_csv(os.path.join(base_dir, 'test.csv'))

print(train.shape,test.shape)



train.head()
noisy_row = [31, 50, 2235, 5244, 10409, 11748, 12384, 14395, 15215, 17629, 20819, 23691, 32089, 39532, 40530, 43954, 48186, 50500, 55834, 60088,

             60442, 61095, 62982, 63803, 67464, 70791, 74861, 73636, 74119, 76275, 79789, 85745, 91058, 91663, 91800, 93204, 99295, 100903, 101177, 103155,

             109166, 109566, 109651, 109724, 110115, 110441, 111461, 113175, 115782, 116903, 118099, 118328, 118414, 119071, 125338, 125340, 129496, 129640, 

             132027, 138212, 131626, 134715, 133248, 136217, 141377, 143707, 145045, 146485, 37301]



train.drop(noisy_row, inplace = True)

train['review_id'] = list(range(train.shape[0]))

train.index = list(range(train.shape[0]))

train
# Dealing with imbalanced data

balance = 'under'

if balance == 'over':

    from imblearn.over_sampling import RandomOverSampler

    oversample = RandomOverSampler(sampling_strategy = 'all')

    X = train.review.values

    y = train.rating.values - 1

    X, y = oversample.fit_resample(X.reshape(-1,1), y)

elif balance == 'under':

    from imblearn.under_sampling import RandomUnderSampler

    undersample = RandomUnderSampler(sampling_strategy = 'all')

    X = train.review.values

    y = train.rating.values - 1

    X, y = undersample.fit_resample(X.reshape(-1,1), y)

else:

    from imblearn.over_sampling import SMOTE

    oversample = SMOTE(sampling_strategy = 'all')

    X = train.review.values

    y = train.rating.values - 1

    

X = X.flatten()

df = {

    'review_id': list(range(len(y))), 

    'review': X, 

    'rating': y

}

train = pd.DataFrame(data = df)

train
# Augmentation with Amazon dataset

train_amazon = pd.read_csv(os.path.join(base_dir, 'amazon_augmented.csv'), index_col = 0)

train = pd.concat((train, train_amazon))

train.review_id = list(range(train.shape[0]))

train.index = list(range(train.shape[0]))

del train_amazon

train
import emoji  # https://pypi.org/project/emoji/



have_emoji_train_idx = []

have_emoji_test_idx = []



for idx, review in enumerate(train['review']):

    if any(char in emoji.UNICODE_EMOJI for char in review):

        have_emoji_train_idx.append(idx)

        

for idx, review in enumerate(test['review']):

    if any(char in emoji.UNICODE_EMOJI for char in review):

        have_emoji_test_idx.append(idx)
def emoji_cleaning(text):

    

    # Change emoji to text

    text = emoji.demojize(text).replace(":", " ")

    

    # Delete repeated emoji

    tokenizer = text.split()

    repeated_list = []

    

    for word in tokenizer:

        if word not in repeated_list:

            repeated_list.append(word)

    

    text = ' '.join(text for text in repeated_list)

    text = text.replace("_", " ").replace("-", " ")

    return text
train_df_original = train.copy()

test_df_original = test.copy()



# emoji_cleaning

train.loc[have_emoji_train_idx, 'review'] = train.loc[have_emoji_train_idx, 'review'].apply(emoji_cleaning)

test.loc[have_emoji_test_idx, 'review'] = test.loc[have_emoji_test_idx, 'review'].apply(emoji_cleaning)
def review_cleaning(text):

    

    # Lowercase and remove newline

    text = text.lower()

    text = re.sub(r'\n', '', text)

    

    # Change emoticon to text

    text = re.sub(r':\(', 'dislike', text)

    text = re.sub(r': \(\(', 'dislike', text)

    text = re.sub(r':, \(', 'dislike', text)

    text = re.sub(r':\)', 'smile', text)

    text = re.sub(r';\)', 'smile', text)

    text = re.sub(r':\)\)\)', 'smile', text)

    text = re.sub(r':\)\)\)\)\)\)', 'smile', text)

    text = re.sub(r'=\)\)\)\)', 'smile', text)

    

    # Remove punctuation

    text = re.sub('[^a-z0-9 ]', ' ', text)

    

    # Remove accented characters from text, e.g. café

    text = unidecode.unidecode(text)

    

    # Expand shortened words, e.g. don't to do not

    for word in text.split():

        if word.lower() in contractions:

            text = text.replace(word, contractions[word.lower()])

    

    # Remove stopwords

    text = ' '.join([word for word in text.split() if word not in STOP_WORDS])

    

    # Numbers to words

    for word in text.split():

        try:

            num2words(word)

            text = text.replace(word, num2words(word))

        except:

            pass

    

    # Lemmatize

    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    

    return text
train['review'] = train['review'].apply(review_cleaning)

test['review'] = test['review'].apply(review_cleaning)
repeated_rows_train = []

repeated_rows_test = []



for idx, review in enumerate(train['review']):

    if re.match(r'\w*(\w)\1+', review):

        repeated_rows_train.append(idx)

        

for idx, review in enumerate(test['review']):

    if re.match(r'\w*(\w)\1+', review):

        repeated_rows_test.append(idx)
def delete_repeated_char(text):

    

    text = re.sub(r'(\w)\1{2,}', r'\1', text)

    

    return text
train.loc[repeated_rows_train, 'review'] = train.loc[repeated_rows_train, 'review'].apply(delete_repeated_char)

test.loc[repeated_rows_test, 'review'] = test.loc[repeated_rows_test, 'review'].apply(delete_repeated_char)
def recover_shortened_words(text):

    

    # put \b (boundary) for avoid the characters in the word to be replaced

    # I only make a few examples here, you can add if you're interested :)

    

    text = re.sub(r'\bapaa\b', 'apa', text)

    

    text = re.sub(r'\bbsk\b', 'besok', text)

    text = re.sub(r'\bbrngnya\b', 'barangnya', text)

    text = re.sub(r'\bbrp\b', 'berapa', text)

    text = re.sub(r'\bbgt\b', 'banget', text)

    text = re.sub(r'\bbngt\b', 'banget', text)

    text = re.sub(r'\bgini\b', 'begini', text)

    text = re.sub(r'\bbrg\b', 'barang', text)

    

    text = re.sub(r'\bdtg\b', 'datang', text)

    text = re.sub(r'\bd\b', 'di', text)

    text = re.sub(r'\bsdh\b', 'sudah', text)

    text = re.sub(r'\bdri\b', 'dari', text)

    text = re.sub(r'\bdsni\b', 'disini', text)

    

    text = re.sub(r'\bgk\b', 'gak', text)

    

    text = re.sub(r'\bhrs\b', 'harus', text)

    

    text = re.sub(r'\bjd\b', 'jadi', text)

    text = re.sub(r'\bjg\b', 'juga', text)

    text = re.sub(r'\bjgn\b', 'jangan', text)

    

    text = re.sub(r'\blg\b', 'lagi', text)

    text = re.sub(r'\blgi\b', 'lagi', text)

    text = re.sub(r'\blbh\b', 'lebih', text)

    text = re.sub(r'\blbih\b', 'lebih', text)

    

    text = re.sub(r'\bmksh\b', 'makasih', text)

    text = re.sub(r'\bmna\b', 'mana', text)

    

    text = re.sub(r'\borg\b', 'orang', text)

    

    text = re.sub(r'\bpjg\b', 'panjang', text)

    

    text = re.sub(r'\bka\b', 'kakak', text)

    text = re.sub(r'\bkk\b', 'kakak', text)

    text = re.sub(r'\bklo\b', 'kalau', text)

    text = re.sub(r'\bkmrn\b', 'kemarin', text)

    text = re.sub(r'\bkmrin\b', 'kemarin', text)

    text = re.sub(r'\bknp\b', 'kenapa', text)

    text = re.sub(r'\bkcil\b', 'kecil', text)

    

    text = re.sub(r'\bgmn\b', 'gimana', text)

    text = re.sub(r'\bgmna\b', 'gimana', text)

    

    text = re.sub(r'\btp\b', 'tapi', text)

    text = re.sub(r'\btq\b', 'thanks', text)

    text = re.sub(r'\btks\b', 'thanks', text)

    text = re.sub(r'\btlg\b', 'tolong', text)

    text = re.sub(r'\bgk\b', 'tidak', text)

    text = re.sub(r'\bgak\b', 'tidak', text)

    text = re.sub(r'\bgpp\b', 'tidak apa apa', text)

    text = re.sub(r'\bgapapa\b', 'tidak apa apa', text)

    text = re.sub(r'\bga\b', 'tidak', text)

    text = re.sub(r'\btgl\b', 'tanggal', text)

    text = re.sub(r'\btggl\b', 'tanggal', text)

    text = re.sub(r'\bgamau\b', 'tidak mau', text)

    

    text = re.sub(r'\bsy\b', 'saya', text)

    text = re.sub(r'\bsis\b', 'sister', text)

    text = re.sub(r'\bsdgkan\b', 'sedangkan', text)

    text = re.sub(r'\bmdh2n\b', 'semoga', text)

    text = re.sub(r'\bsmoga\b', 'semoga', text)

    text = re.sub(r'\bsmpai\b', 'sampai', text)

    text = re.sub(r'\bnympe\b', 'sampai', text)

    text = re.sub(r'\bdah\b', 'sudah', text)

    

    text = re.sub(r'\bberkali2\b', 'repeated', text)

    

    text = re.sub(r'\byg\b', 'yang', text)

    

    return text
train['review'] = train['review'].apply(recover_shortened_words)

test['review'] = test['review'].apply(recover_shortened_words)
train
MODEL_CLASSES = {

    'bert': (BertForSequenceClassification, BertTokenizer, BertConfig),

    'xlnet': (XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig),

    'xlm': (XLMForSequenceClassification, XLMTokenizer, XLMConfig),

    'roberta': (RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig),

    'distilbert': (DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig)

}
# Parameters

seed = 42

use_fp16 = False

bs = 16



model_type = 'roberta'

pretrained_model_name = 'roberta-base'



# model_type = 'bert'

# pretrained_model_name='bert-base-uncased'



# model_type = 'distilbert'

# pretrained_model_name = 'distilbert-base-uncased'



#model_type = 'xlm'

#pretrained_model_name = 'xlm-clm-enfr-1024'



# model_type = 'xlnet'

# pretrained_model_name = 'xlnet-base-cased'
model_class, tokenizer_class, config_class = MODEL_CLASSES[model_type]
def seed_all(seed_value):

    random.seed(seed_value) # Python

    np.random.seed(seed_value) # cpu vars

    torch.manual_seed(seed_value) # cpu  vars

    

    if torch.cuda.is_available(): 

        torch.cuda.manual_seed(seed_value)

        torch.cuda.manual_seed_all(seed_value) # gpu vars

        torch.backends.cudnn.deterministic = True  #needed

        torch.backends.cudnn.benchmark = False
seed_all(seed)
class TransformersBaseTokenizer(BaseTokenizer):

    """Wrapper around PreTrainedTokenizer to be compatible with fast.ai"""

    def __init__(self, pretrained_tokenizer: PreTrainedTokenizer, model_type = 'bert', **kwargs):

        self._pretrained_tokenizer = pretrained_tokenizer

        self.max_seq_len = pretrained_tokenizer.max_len

        self.model_type = model_type



    def __call__(self, *args, **kwargs): 

        return self



    def tokenizer(self, t:str) -> List[str]:

        """Limits the maximum sequence length and add the spesial tokens"""

        CLS = self._pretrained_tokenizer.cls_token

        SEP = self._pretrained_tokenizer.sep_token

        if self.model_type in ['roberta']:

            tokens = self._pretrained_tokenizer.tokenize(t, add_prefix_space=True)[:self.max_seq_len - 2]

            tokens = [CLS] + tokens + [SEP]

        else:

            tokens = self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2]

            if self.model_type in ['xlnet']:

                tokens = tokens + [SEP] +  [CLS]

            else:

                tokens = [CLS] + tokens + [SEP]

        return tokens
transformer_tokenizer = tokenizer_class.from_pretrained(pretrained_model_name)

transformer_base_tokenizer = TransformersBaseTokenizer(pretrained_tokenizer = transformer_tokenizer, model_type = model_type)

fastai_tokenizer = Tokenizer(tok_func = transformer_base_tokenizer, pre_rules = [], post_rules = [])
class TransformersVocab(Vocab):

    def __init__(self, tokenizer: PreTrainedTokenizer):

        super(TransformersVocab, self).__init__(itos = [])

        self.tokenizer = tokenizer

    

    def numericalize(self, t:Collection[str]) -> List[int]:

        "Convert a list of tokens `t` to their ids."

        return self.tokenizer.convert_tokens_to_ids(t)

        #return self.tokenizer.encode(t)



    def textify(self, nums:Collection[int], sep=' ') -> List[str]:

        "Convert a list of `nums` to their tokens."

        nums = np.array(nums).tolist()

        return sep.join(self.tokenizer.convert_ids_to_tokens(nums)) if sep is not None else self.tokenizer.convert_ids_to_tokens(nums)

    

    def __getstate__(self):

        return {'itos':self.itos, 'tokenizer':self.tokenizer}



    def __setstate__(self, state:dict):

        self.itos = state['itos']

        self.tokenizer = state['tokenizer']

        self.stoi = collections.defaultdict(int,{v:k for k,v in enumerate(self.itos)})
transformer_vocab =  TransformersVocab(tokenizer = transformer_tokenizer)

numericalize_processor = NumericalizeProcessor(vocab = transformer_vocab)



tokenize_processor = TokenizeProcessor(tokenizer = fastai_tokenizer, include_bos = False, include_eos = False)



transformer_processor = [tokenize_processor, numericalize_processor]
pad_first = bool(model_type in ['xlnet'])

pad_idx = transformer_tokenizer.pad_token_id
tokens = transformer_tokenizer.tokenize(train.review[0])

print(tokens)

ids = transformer_tokenizer.convert_tokens_to_ids(tokens)

print(ids)

transformer_tokenizer.convert_ids_to_tokens(ids)
train.to_csv('train_clean.csv')

test.to_csv('test_clean.csv')



# Load

path = '/kaggle/working'

train = pd.read_csv(os.path.join(path, 'train_clean.csv'), index_col = 0)

test = pd.read_csv(os.path.join(path, 'test_clean.csv'), index_col = 0)



train = train.dropna()
print(train.shape, test.shape)
# Split training and validation set

databunch = (TextList.from_df(train, cols = ['review'], processor = transformer_processor)

             .split_by_rand_pct(0.2, seed = seed)

             .label_from_df(cols = ['rating'])

             .add_test(test)

             .databunch(bs = bs, pad_first = pad_first, pad_idx = pad_idx))
print('[CLS] token :', transformer_tokenizer.cls_token)

print('[SEP] token :', transformer_tokenizer.sep_token)

print('[PAD] token :', transformer_tokenizer.pad_token)

databunch.show_batch()
# Encode tokens

print('[CLS] id :', transformer_tokenizer.cls_token_id)

print('[SEP] id :', transformer_tokenizer.sep_token_id)

print('[PAD] id :', pad_idx)

test_one_batch = databunch.one_batch()[0]

print('Batch shape : ',test_one_batch.shape)

print(test_one_batch)
# defining our model architecture 

class CustomTransformerModel(nn.Module):

    def __init__(self, transformer_model: PreTrainedModel):

        super(CustomTransformerModel,self).__init__()

        self.transformer = transformer_model

        

    def forward(self, input_ids, attention_mask = None):

        

        # attention_mask

        # Mask to avoid performing attention on padding token indices.

        # Mask values selected in ``[0, 1]``:

        # ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

        attention_mask = (input_ids != pad_idx).type(input_ids.type()) 

        

        logits = self.transformer(input_ids,

                                  attention_mask = attention_mask)[0]   

        return logits
# Set configuration for the model

config = config_class.from_pretrained(pretrained_model_name)

config.num_labels = 5

config.use_bfloat16 = use_fp16

print(config)
transformer_model = model_class.from_pretrained(pretrained_model_name, config = config)

custom_transformer_model = CustomTransformerModel(transformer_model = transformer_model)
from fastai.callbacks import *

from transformers import AdamW

from functools import partial



CustomAdamW = partial(AdamW, correct_bias = False)



learner = Learner(databunch, 

                  custom_transformer_model, 

                  opt_func = CustomAdamW, 

                  metrics = [accuracy, error_rate])



# Show graph of learner stats and metrics after each epoch.

learner.callbacks.append(ShowGraph(learner))



# Put learn in FP16 precision mode. --> Seems to not working

if use_fp16: learner = learner.to_fp16()
print(learner.model)
# For DistilBERT

# list_layers = [learner.model.transformer.distilbert.embeddings,

#                learner.model.transformer.distilbert.transformer.layer[0],

#                learner.model.transformer.distilbert.transformer.layer[1],

#                learner.model.transformer.distilbert.transformer.layer[2],

#                learner.model.transformer.distilbert.transformer.layer[3],

#                learner.model.transformer.distilbert.transformer.layer[4],

#                learner.model.transformer.distilbert.transformer.layer[5],

#                learner.model.transformer.pre_classifier]



# For xlnet-base-cased

# list_layers = [learner.model.transformer.transformer.word_embedding,

#               learner.model.transformer.transformer.layer[0],

#               learner.model.transformer.transformer.layer[1],

#               learner.model.transformer.transformer.layer[2],

#               learner.model.transformer.transformer.layer[3],

#               learner.model.transformer.transformer.layer[4],

#               learner.model.transformer.transformer.layer[5],

#               learner.model.transformer.transformer.layer[6],

#               learner.model.transformer.transformer.layer[7],

#               learner.model.transformer.transformer.layer[8],

#               learner.model.transformer.transformer.layer[9],

#               learner.model.transformer.transformer.layer[10],

#               learner.model.transformer.transformer.layer[11],

#               learner.model.transformer.sequence_summary]



# For roberta-base

list_layers = [learner.model.transformer.roberta.embeddings,

              learner.model.transformer.roberta.encoder.layer[0],

              learner.model.transformer.roberta.encoder.layer[1],

              learner.model.transformer.roberta.encoder.layer[2],

              learner.model.transformer.roberta.encoder.layer[3],

              learner.model.transformer.roberta.encoder.layer[4],

              learner.model.transformer.roberta.encoder.layer[5],

              learner.model.transformer.roberta.encoder.layer[6],

              learner.model.transformer.roberta.encoder.layer[7],

              learner.model.transformer.roberta.encoder.layer[8],

              learner.model.transformer.roberta.encoder.layer[9],

              learner.model.transformer.roberta.encoder.layer[10],

              learner.model.transformer.roberta.encoder.layer[11],

              learner.model.transformer.roberta.pooler]
learner.split(list_layers)

num_groups = len(learner.layer_groups)

print('Learner split in',num_groups,'groups')

print(learner.layer_groups)
# Load untrained model

learner.save('untrain')



seed_all(seed)

learner.load('untrain')
# Optimal learning rate

learner.freeze_to(-1)

learner.summary()

learner.lr_find()

learner.recorder.plot(skip_end = 10, suggestion = True)
# First cycle

learner.fit_one_cycle(5, max_lr = 3.98E-05, moms = (0.8,0.7))

learner.save('first_cycle')



# Second cycle

seed_all(seed)

learner.load('first_cycle')

learner.freeze_to(-2)

lr = 1e-5

learner.fit_one_cycle(5, max_lr = slice(lr*0.95**num_groups, lr), moms = (0.8, 0.9))

learner.save('second_cycle')



# Third cycle

seed_all(seed)

learner.load('second_cycle')

learner.unfreeze()

learner.fit_one_cycle(5, max_lr = slice(lr*0.95**num_groups, lr), moms = (0.8, 0.9))

learner.save('third_cycle')



r'''# Fourth

seed_all(seed)

learner.load('third_cycle')

learner.freeze_to(-4)

learner.fit_one_cycle(10, max_lr = slice(lr*0.95**num_groups, lr), moms = (0.8, 0.9))

learner.save('fourth_cycle')



# Fifth cycle

seed_all(seed)

learner.load('fourth_cycle')

learner.unfreeze()

learner.fit_one_cycle(10, max_lr = slice(lr*0.95**num_groups, lr), moms = (0.8, 0.9))

learner.save('fifth_cycle')'''
r'''# Continue training

# Fourth cycle

learner.load(os.path.join(base_dir, 'third_cycle'))



learner.unfreeze()



learner.fit_one_cycle(10, max_lr = slice(lr*0.95**num_groups, lr), moms = (0.8, 0.9))



learner.save('fourth_cycle')



# Fifth cycle

seed_all(seed)

learner.load('fourth_cycle')



learner.unfreeze()



learner.fit_one_cycle(10, max_lr = slice(lr*0.95**num_groups, lr), moms = (0.8, 0.9))



learner.save('fifth_cycle')



# Sixth cycle

seed_all(seed)

learner.load('fifth_cycle')



learner.unfreeze()



learner.fit_one_cycle(10, max_lr = slice(lr*0.95**num_groups, lr), moms = (0.8, 0.9))



learner.save('sixth_cycle')'''
learner.predict('The product quality is super good')
learner.predict('Delay delivery, bad quality')
learner.export(file = 'transformer.pkl')
path = '/kaggle/working'

export_learner = load_learner(path, file = 'transformer.pkl')
export_learner.predict('Delay delivery, bad quality')
# Prediction

def get_preds_as_nparray(ds_type) -> np.ndarray:

    """

    the get_preds method does not yield the elements in order by default

    we borrow the code from the RNNLearner to resort the elements into their correct order

    """

    preds = learner.get_preds(ds_type)[0].detach().cpu().numpy()

    sampler = [i for i in databunch.dl(ds_type).sampler]

    reverse_sampler = np.argsort(sampler)

    return preds[reverse_sampler, :]



test_preds = get_preds_as_nparray(DatasetType.Test)

np.save('test_preds.npy', test_preds)
sample_submission = pd.read_csv(os.path.join(base_dir, 'test.csv'))

sample_submission['rating'] = np.argmax(test_preds, axis = 1) + 1

sample_submission = sample_submission.drop(['review'], axis = 1)

sample_submission.to_csv("predictions.csv", index = False)