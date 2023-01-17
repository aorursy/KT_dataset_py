import os

import sys

import gc

import random

import re

import string

from tqdm.notebook import tqdm

import numpy as np



sys.path.extend(['../input/transformer/', '../input/sacremoses/sacremoses-master/'])



import pandas as pd

from bs4 import BeautifulSoup

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import KFold

from scipy.stats import spearmanr

    

import torch

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, Dataset

from torch import nn

from torch.nn import Module

from torch.nn import functional as f



import transformers

from transformers import BertTokenizer, BertConfig, BertModel, XLNetConfig, XLNetModel

from transformers import RobertaConfig, RobertaModel, DistilBertModel, DistilBertConfig

from transformers.tokenization_bert import BasicTokenizer, whitespace_tokenize

from transformers.optimization import AdamW, get_linear_schedule_with_warmup



from nltk.corpus import stopwords



stop_word = set(stopwords.words('english'))

print(len(stop_word))



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)
csv_path = '../input/nlp-getting-started/'

train_path = csv_path + 'train.csv'

test_path = csv_path + 'test.csv'

submission_path = csv_path + 'sample_submission.csv'



path_model = '../input/pretrained-bert-models-for-pytorch/'

model_file = path_model + 'bert-base-uncased/pytorch_model.bin'

config_file = path_model + 'bert-base-uncased/bert_config.json'

vocab_file = path_model + 'bert-base-uncased-vocab.txt'



path_model = '../input/pretrained-bert-models-for-pytorch/'

model_file_large = path_model + 'bert-large-uncased/pytorch_model.bin'

config_file_large = path_model + 'bert-large-uncased/bert_config.json'

vocab_file_large = path_model + 'bert-large-uncased-vocab.txt'



path_roberta = '../input/roberta-transformers-pytorch/roberta-base/'

config_roberta = path_roberta + 'config.json'

vocab_roberta = path_roberta + 'vocab.json'

model_roberta = path_roberta + 'pytorch_model.bin'



path_distilroberta = '../input/roberta-transformers-pytorch/distilroberta-base/'

config_distilroberta = path_distilroberta + 'config.json'

vocab_distilroberta = path_distilroberta + 'vocab.json'

model_distilroberta = path_distilroberta + 'pytorch_model.bin'



path_xlnet = '../input/xlnet-pretrained-models-pytorch/'

config_xlnet = path_xlnet + 'xlnet-base-cased-config.json'

vocab_xlnet = path_model + 'bert-base-uncased-vocab.txt'

model_xlnet = path_xlnet + 'xlnet-base-cased-pytorch_model.bin'



model_use = 'bert'

num_model = 7
tokenize = BertTokenizer.from_pretrained(vocab_file, do_lower_case=True, do_basic_tokenize=True)
train_csv = pd.read_csv(train_path)

train_csv = train_csv[['id', 'text', 'target']]

train_csv.head(10)
test_csv = pd.read_csv(test_path)

test_csv = test_csv[['id', 'text']]

test_csv['target'] = 0

test_csv.head(10)
%%time

text_length = test_csv['text'].apply(lambda x: len(tokenize.tokenize(x)))



plt.figure(figsize=(10, 8))

sns.distplot(text_length)

print(f'max lenth of text: {max(text_length)}')
submission = pd.read_csv(submission_path)

submission.head()
#clean data

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',

 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '\xa0', '\t',

 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '\u3000', '\u202f',

 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '«',

 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]



mispell_dict = {"aren't" : "are not",

"can't" : "cannot",

"couldn't" : "could not",

"couldnt" : "could not",

"didn't" : "did not",

"doesn't" : "does not",

"doesnt" : "does not",

"don't" : "do not",

"hadn't" : "had not",

"hasn't" : "has not",

"haven't" : "have not",

"havent" : "have not",

"he'd" : "he would",

"he'll" : "he will",

"he's" : "he is",

"i'd" : "I would",

"i'd" : "I had",

"i'll" : "I will",

"i'm" : "I am",

"isn't" : "is not",

"it's" : "it is",

"it'll":"it will",

"i've" : "I have",

"let's" : "let us",

"mightn't" : "might not",

"mustn't" : "must not",

"shan't" : "shall not",

"she'd" : "she would",

"she'll" : "she will",

"she's" : "she is",

"shouldn't" : "should not",

"shouldnt" : "should not",

"that's" : "that is",

"thats" : "that is",

"there's" : "there is",

"theres" : "there is",

"they'd" : "they would",

"they'll" : "they will",

"they're" : "they are",

"theyre":  "they are",

"they've" : "they have",

"we'd" : "we would",

"we're" : "we are",

"weren't" : "were not",

"we've" : "we have",

"what'll" : "what will",

"what're" : "what are",

"what's" : "what is",

"what've" : "what have",

"where's" : "where is",

"who'd" : "who would",

"who'll" : "who will",

"who're" : "who are",

"who's" : "who is",

"who've" : "who have",

"won't" : "will not",

"wouldn't" : "would not",

"you'd" : "you would",

"you'll" : "you will",

"you're" : "you are",

"you've" : "you have",

"'re": " are",

"wasn't": "was not",

"we'll":" will",

"didn't": "did not"}



puncts = puncts + list(string.punctuation)



def clean_text(x):

    x = str(x).replace("\n","")

    for punct in puncts:

        x = x.replace(punct, f' {punct} ')

    return x





def clean_numbers(x):

    x = re.sub('\d+', ' ', x)

    return x





def replace_typical_misspell(text):

    mispellings_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))



    def replace(match):

        return mispell_dict[match.group(0)]



    return mispellings_re.sub(replace, text)



def remove_space(string):

    string = BeautifulSoup(string).text.strip().lower()

    string = re.sub(r'((http)\S+)', 'http', string)

    string = re.sub(r'\s+', ' ', string)

    return string





def clean_data(df, columns: list):

    

    for col in columns:

        df[col] = df[col].apply(lambda x: remove_space(x).lower())        

        df[col] = df[col].apply(lambda x: replace_typical_misspell(x))

        df[col] = df[col].apply(lambda x: clean_text(x))

        

    return df
%%time

test = test_csv.loc[:, ['id', 'text']]

train = train_csv.loc[:, ['text', 'target']]



test = clean_data(test, ['text'])

train = clean_data(train, ['text'])

test.head()
class QueryDataset(Dataset):

    

    def __init__(self, data, is_train=True, max_length=512):

        

        super(QueryDataset, self).__init__()

        

        self.max_length = max_length

        self.data = data

        self.is_train = is_train

        self.tokenizer = BertTokenizer.from_pretrained(vocab_file_large, do_lower_case=True, do_basic_tokenize=True)

    

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, idx):

        token_ids = self.get_token_ids(idx)

        

        if self.is_train:

            label = torch.tensor(self.data.loc[idx, 'target'], dtype=torch.float32)

            return token_ids, label

        else:

            return token_ids

        

    

    def get_token_ids(self, idx):

        

        token = self.tokenizer.tokenize(self.data.loc[idx, 'text'])

        

        max_seq_length = self.max_length - 2        

        if len(token) > max_seq_length:           

            token = token[:max_seq_length]                                                             

                                                         

        token = ['[CLS]'] + token + ['[SEP]']

        token_ids_org = self.tokenizer.convert_tokens_to_ids(token)

       

        if len(token_ids_org) < self.max_length:

            token_ids = token_ids_org + [0]*(self.max_length - len(token_ids_org))

        else:

            token_ids = token_ids_org[:self.max_length]

            

        token_ids = torch.tensor(token_ids)

        del token_ids_org

        return token_ids

                

    def collate_fn(self, batch):

                

        if self.is_train:

            token_ids = torch.stack([x[0] for x in batch])

            label = torch.stack([x[1] for x in batch])

            return token_ids, label

        else:

            token_ids = torch.stack([x for x in batch])

            return token_ids
test_dataset = QueryDataset(test, is_train=False)

test_ld = DataLoader(test_dataset, batch_size=8,

                     shuffle=False, num_workers=0, collate_fn=test_dataset.collate_fn)



print(len(test_ld))
class BertLinear(Module):

    

    def __init__(self, model_name, max_length=512, num_class=1):

        super(BertLinear, self).__init__()

        

        if model_name == 'bert':            

            config = BertConfig.from_json_file(config_file)

            self.bert = BertModel.from_pretrained(model_file, config=config)

            

        elif model_name == 'bert-large':

            config = BertConfig.from_json_file(config_file_large)

            self.bert = BertModel.from_pretrained(model_file_large, config=config)

            

        elif model_name == 'robert':

            config = RobertaConfig()

            self.bert = RobertaModel(config=config)

            

        elif model_name == 'distilrobert':

            config = DistilBertConfig()

            self.bert = DistilBertModel(config=config)

            

        elif model_name == 'xlnet':

            config = XLNetConfig()

            self.bert = XLNetModel(config=config)

                                               

        self.bert.config.max_position_embeddings=max_length            

        self.dropout = nn.Dropout(0.2)

        self.fc = nn.Sequential(nn.ReLU(inplace=True),

                                nn.Linear(3*config.hidden_size, num_class))

    

    def forward(self, input_ids, segment_ids=None):

        

        attention_mask = (input_ids > 0).float()

        segment_ids = torch.zeros_like(input_ids)

        layer, pooler = self.bert(input_ids=input_ids,

                                  attention_mask=attention_mask,

                                  token_type_ids=segment_ids)

        

        avg_pool = torch.mean(layer, 1)

        max_pool, _ = torch.max(layer, 1)

    

        pooler = torch.cat((avg_pool, max_pool, pooler), 1)

        

        output = self.dropout(pooler)  

        logits = self.fc(output)

        

        return logits

    



def load_model(model_name, path_model, load_weight=True):

    models = []    

    model = BertLinear(model_name).to(device)

    

    if load_weight:

        for weight in sorted(os.listdir(path_model)):

            if 'pth' in weight:

                weight_path = os.path.join(path_model, weight)

                state = torch.load(weight_path, map_location=lambda storage, loc: storage)

                models.append(state)

    else:

        for i in range(num_model):            

            models.append(model.state_dict())

        

    return models



base_model = load_model(model_use, path_model='../input/tweet-bert')
def loss_fn(pred, expected):

    return 0.7*f.mse_loss(torch.sigmoid(pred), expected) + 0.3*f.binary_cross_entropy_with_logits(pred, expected)
class Trainer(object):

    

    def __init__(self, base_model, model_name='bert',

                 weight_decay=0.1, learning_rate=2e-5):

        

        self.learning_rate = learning_rate

        self.weight_decay = weight_decay

        

        self.model_name = model_name

        self.base_model = base_model

        self.cretion = loss_fn

    

    def train(self, folds, epochs, train, check_number=5):

        

        model = BertLinear(self.model_name).to(device)

        score_val_max = [0]*folds

        for fold, (train_index, val_index) in enumerate(KFold(n_splits=folds, shuffle=True, random_state=37).split(train)):

            print(f'fold: {fold}')

            val_score_max = 0

        

            train_df = train.iloc[train_index]

            train_df.reset_index(inplace=True, drop=True)

            

            val_df = train.iloc[val_index]

            val_df.reset_index(inplace=True, drop=True)

            

            model.load_state_dict(self.base_model[fold])

            

            optimizer = AdamW(model.parameters(),

                              lr=self.learning_rate,

                              weight_decay=self.weight_decay,

                              correct_bias=False)            

            

            train_dataset = QueryDataset(train_df)

            train_ld = DataLoader(train_dataset, batch_size=8, shuffle=True,

                                  num_workers=0, collate_fn=train_dataset.collate_fn)

            

            val_dataset = QueryDataset(val_df)

            val_ld = DataLoader(val_dataset, batch_size=8, shuffle=True,

                                num_workers=0, collate_fn=val_dataset.collate_fn)

            

            schedule = get_linear_schedule_with_warmup(optimizer,

                                                       num_warmup_steps=0.5,

                                                       num_training_steps=epochs*len(val_ld))

            

            del val_dataset, train_dataset, val_df, train_df

            model.zero_grad()

            check_score = 0

            for epoch in range(epochs):

                print(f'Epoch: {epoch}')

                train_loss = 0

                val_loss = 0



                model.train()

                for token_ids, label in tqdm(train_ld):



                    optimizer.zero_grad()

                    token_ids, label = token_ids.to(device), label.unsqueeze(1).to(device)

                    output = model(token_ids)

                    loss = self.cretion(output, label)

                    loss.backward()



                    train_loss += loss.item()

                    optimizer.step()

                    schedule.step()

                    del token_ids, label

                    

                train_loss = train_loss/len(train_ld)

                torch.cuda.empty_cache()

                gc.collect()

                

                # evaluate process

                model.eval()

                score_val = 0

                with torch.no_grad():

                    for token_ids, label in tqdm(val_ld):

                        token_ids, label = token_ids.to(device), label.unsqueeze(1).to(device)



                        output = model(token_ids)

                        loss = self.cretion(output, label)

                        score_val += torch.sum((torch.sigmoid(output) >= 0.5).float()==label).item()/output.size(0)

                        val_loss += loss.item()

                    

                    score_val = score_val/len(val_ld)

                    val_loss = val_loss/len(val_ld)             

                    

                    

                print(f'train_loss: {train_loss:.4f}, valid_loss: {val_loss:.4f}, valid_score: {score_val:.4f}')

                schedule.step(val_loss)



                if score_val >= val_score_max:

                    score_val_max[fold] = score_val

                    check_score+=1

                    print(f'Validation score increased ({val_score_max:.4f} --> {score_val:.4f}). Saving model...')

                    val_score_max = score_val

                    check_score = 0

                    torch.save(model.state_dict(), f'model_fold_{str(fold)}.pth')

                else:

                    check_score += 1

                    print(f'{check_score} epochs of decreasing val_score')



                    if check_score > check_number:

                        print('Stopping trainning!')                    

                        break

                        

            del optimizer, schedule, train_ld, val_ld

            torch.cuda.empty_cache()

            

            gc.collect()

        

        return score_val_max

            

    def predict(self, test_ld, submission, threshold=0.5):

        

        model = BertLinear(self.model_name).to(device)  

        list_predict = []

        

        for token_ids in tqdm(test_ld):

            predicts = []

            for index_model, model_param in enumerate(self.base_model):

                model.load_state_dict(model_param)

                

                model.eval()

                with torch.no_grad():

                    token_ids = token_ids.to(device)                    

                    predict_prob = torch.sigmoid(model(token_ids))

                    predict = (predict_prob>threshold).float().cpu().numpy()

                    

                predicts.append(predict) 

                

            predicts = np.sum(predicts, axis=0)

            list_predict.extend(np.where(predicts>3, 1, 0))

                

        submission.target = np.array(list_predict).flatten()

            

        return submission        
train_process = Trainer(base_model, model_use)



test_csv = train_process.predict(test_ld=test_ld, submission=test_csv, threshold=0.5)

train_csv = pd.concat([train_csv, test_csv]).sort_values('id')



train_csv.head()
train = train_csv.loc[:, ['text', 'target']]

train = clean_data(train, ['text'])



del train_csv, test_csv
score_val_max = train_process.train(folds=num_model,

                                    epochs=3,

                                    train=train,

                                    check_number=3)





del base_model, train



torch.cuda.empty_cache()

gc.collect()
base_model1 = load_model(model_use, path_model='.')
train_process = Trainer(base_model1, model_use)

submission = train_process.predict(test_ld=test_ld, submission=submission, threshold=0.5)



submission.to_csv('submission.csv', index=False)

submission.head(20)