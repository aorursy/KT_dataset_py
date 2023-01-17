import os

import sys

import gc

import random

import re

from tqdm.notebook import tqdm

import numpy as np

from collections import deque



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



import transformers

from transformers import BertTokenizer, BertConfig, BertPreTrainedModel, BertModel

from transformers import AlbertTokenizer, AlbertConfig, AlbertModel, AlbertPreTrainedModel

from transformers.tokenization_bert import BasicTokenizer, whitespace_tokenize

from transformers.optimization import AdamW, get_linear_schedule_with_warmup



from nltk.corpus import stopwords



stop_word = set(stopwords.words('english'))

print(len(stop_word))



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)
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

"didn't": "did not",

"tryin'":"trying"}





def clean_text(x):

    x = str(x).replace("\n","")

    for punct in puncts:

        x = x.replace(punct, ' '+punct+' ')

    return x





def clean_numbers(x):

    x = re.sub('[0-9]{5,}', '#####', x)

    x = re.sub('[0-9]{4}', '####', x)

    x = re.sub('[0-9]{3}', '###', x)

    x = re.sub('[0-9]{2}', '##', x)

    return x





def _get_mispell(mispell_dict):

    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))

    return mispell_dict, mispell_re





def replace_typical_misspell(text):

    mispellings, mispellings_re = _get_mispell(mispell_dict)



    def replace(match):

        return mispellings[match.group(0)]



    return mispellings_re.sub(replace, text)



def remove_space(string):

    string = BeautifulSoup(string).text.strip().lower()

    string = re.sub(r'\s+', ' ', string)

    string = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' url ', string)

    return string





def clean_data(df, columns: list):

    

    for col in columns:

        df[col] = df[col].apply(lambda x: remove_space(x).lower())        

        df[col] = df[col].apply(lambda x: replace_typical_misspell(x))

        df[col] = df[col].apply(lambda x: clean_text(x))

        

    return df
path = '../input/google-quest-challenge/'

train_path = path + 'train.csv'

test_path = path + 'test.csv'

submission_path = path + 'sample_submission.csv'



path_model = '../input/pretrained-bert-models-for-pytorch/'

path_albert = '../input/albert-large/albert-large-pytorch_model.bin'

config_albert = '../input/albert-large/albert-large-config.json'



model_file = path_model + 'bert-base-uncased/pytorch_model.bin'

config_file = path_model + 'bert-base-uncased/bert_config.json'

vocab_file = path_model + 'bert-base-uncased-vocab.txt'
train = pd.read_csv(train_path)

print(train.columns)

train.tail()
test = pd.read_csv(test_path)

print(test.columns)

test.tail()
submission = pd.read_csv(submission_path)

target = submission.columns[1:].to_list()

print(len(target))



input_columns = ['question_title', 'question_body', 'answer']
%%time



train = clean_data(train, input_columns)

test = clean_data(test, input_columns)
question_title = train.question_title.apply(lambda x:len(x.split(' ')))

print(max(question_title))

plt.figure(figsize=(10, 8))

sns.distplot(question_title)
question_body = train.question_body.apply(lambda x:len(x.split(' ')))



plt.figure(figsize=(10, 8))

sns.distplot(question_body)
answer = train.answer.apply(lambda x:len(x.split(' ')))



plt.figure(figsize=(10, 8))

sns.distplot(answer)
class QueryDataset(Dataset):

    

    def __init__(self, data, is_train=True, max_query_title = 46, max_length=512):

        

        super(QueryDataset, self).__init__()

        

        self.max_length = max_length

        self.max_query_title = max_query_title

        self.data = data

        self.is_train = is_train

        self.tokenizer = BertTokenizer.from_pretrained(vocab_file, do_lower_case=True, do_basic_tokenize=True)

    

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, idx):

        token_ids, segment_ids = self.get_token_ids(idx)

        

        if self.is_train:

            label = torch.tensor(self.data.loc[idx, target], dtype=torch.float32)

            return token_ids, segment_ids, label

        else:

            return token_ids, segment_ids

        

    

    def get_token_ids(self, idx):

        

        t = self.tokenizer.tokenize(self.data.loc[idx, input_columns[0]])

        b = self.tokenizer.tokenize(self.data.loc[idx, input_columns[1]])

        a = self.tokenizer.tokenize(self.data.loc[idx, input_columns[2]])

        

        t_len, b_len, a_len = len(t), len(b), len(a)

        all_len = t_len + b_len + a_len + 4

        max_query_body = (self.max_length - self.max_query_title - 4)//2

        max_seq_length = self.max_length - 4 - self.max_query_title - max_query_body

        

        if all_len > self.max_length:            

            if t_len < self.max_query_title:

                t_new_len = t_len

                max_query_body = (self.max_length - t_len - 4)//2

                max_seq_length = self.max_length - 4 - t_new_len - max_query_body

            else:

                t_new_len = self.max_query_title

                

            if a_len < max_seq_length:

                a_new_len = a_len

                b_new_len = max_query_body + (max_seq_length - a_len)

            elif b_len < max_query_body:                

                a_new_len = max_seq_length + (max_query_body - b_len)

                b_new_len = b_len

            else:

                a_new_len = max_seq_length

                b_new_len = max_query_body

        else:

            t_new_len, b_new_len, a_new_len = t_len, b_len, a_len

                                                 

                                                         

        token = ['[CLS]'] + t[:t_new_len] + ['[SEP]'] + b[:b_new_len] + ['[SEP]'] + a[:a_new_len] + ['[SEP]']

        token_ids_org = self.tokenizer.convert_tokens_to_ids(token)

       

        if len(token_ids_org) < self.max_length:

            token_ids = token_ids_org + [0]*(self.max_length - len(token_ids_org))

        else:

            token_ids = token_ids_org[:self.max_length]

            

        token_ids = torch.tensor(token_ids)

        segment_ids = [1]*len(token_ids_org)        

        

        sep_one = True

        for index, tk in enumerate(token_ids_org): 

            segment_ids[index] = 0

            if tk == 102:

                if sep_one:

                    sep_one = False

                else:

                    break                    

        

        segment_ids += [0]*(self.max_length - len(token_ids_org))

        segment_ids = torch.tensor(segment_ids)

        del token_ids_org

        

        return token_ids, segment_ids

                

    def collate_fn(self, batch):

        

        token_ids = torch.stack([x[0] for x in batch])

        segment_ids = torch.stack([x[1] for x in batch])

        

        if self.is_train:

            label = torch.stack([x[2] for x in batch])

            return token_ids, segment_ids, label

        else:

            return token_ids, segment_ids
config = BertConfig.from_json_file(config_file)

config
def spearmanr_score(expected, pred):

    score = deque()

    expected, pred = expected.cpu().detach().numpy(), pred.cpu().detach().numpy()

    for i in range(pred.shape[1]):

        score.append(np.nan_to_num(spearmanr(expected[:, i], pred[:, i]).correlation))

    

    return np.mean(score)
class BertLinear(BertPreTrainedModel):

    

    def __init__(self, config, num_class):

        super(BertLinear, self).__init__(config)

        

        self.bert = BertModel.from_pretrained(model_file, config=config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.fc = nn.Linear(config.hidden_size, num_class)

    

    def forward(self, input_ids, segment_ids=None):

        attention_mask = (input_ids > 1).float()

        layer, pooler = self.bert(input_ids=input_ids,

                                  attention_mask=attention_mask,

                                  token_type_ids=segment_ids)

        output = self.dropout(pooler)  

        logits = self.fc(output)

        

        return logits

    



def load_model(load_weight=True):

    

    models = []

    config = BertConfig.from_json_file(config_file)

    model = BertLinear(config, len(target)).to(device)

    model = nn.DataParallel(model)

    

    if load_weight:

        for weight in sorted(os.listdir('../input/train-bert')):

            if 'pth' in weight:

                weight_path = os.path.join('../input/train-bert', weight)

                state = torch.load(weight_path, map_location=lambda storage, loc: storage)

                models.append(state)

    else:

        model.to(device)

        for i in range(5):            

            models.append(model.state_dict())

        

    return models



base_model = load_model(True)
class Trainer(object):

    

    def __init__(self, config, base_model,

                 weight_decay=0.1, learning_rate=2e-5):

        

        self.learning_rate = learning_rate

        self.weight_decay = weight_decay

        

        self.config = config

        self.base_model = base_model

        self.cretion = nn.MSELoss()

        self.score = spearmanr_score

    

    def train(self, folds, epochs, train, check_number=5):

        

        model = BertLinear(config, len(target)).to(device)

        model = nn.DataParallel(model)

        

        for fold, (train_index, val_index) in enumerate(KFold(n_splits=folds, shuffle=True).split(train)):

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

                                                       num_warmup_steps=0.05,

                                                       num_training_steps=epochs*len(train_ld))

            

            del val_dataset, train_dataset, val_df, train_df

            model.zero_grad()

            check_score = 0

            for epoch in range(epochs):

                print(f'Epoch: {epoch}')

                train_loss = 0

                val_loss = 0



                model.train()

                for token_ids, segment_ids, label in tqdm(train_ld):



                    optimizer.zero_grad()

                    token_ids, segment_ids, label = token_ids.to(device), segment_ids.to(device), label.to(device)

                    output = torch.sigmoid(model(token_ids, segment_ids))

                    

                    loss = self.cretion(output, label)

                    loss.backward()

                    train_loss += loss.item()

                    

                    optimizer.step()

                    schedule.step()

                    del token_ids, segment_ids, label

                    

                train_loss = train_loss/len(train_ld)

                torch.cuda.empty_cache()

                gc.collect()

                

                # evaluate process

                model.eval()

                score_val = 0

                with torch.no_grad():

                    for token_ids, segment_ids, label in tqdm(val_ld):

                        token_ids, segment_ids, label = token_ids.to(device), segment_ids.to(device), label.to(device)



                        output = torch.sigmoid(model(token_ids, segment_ids))

                        loss = self.cretion(output, label)

                        score_val += self.score(output, label)

                        val_loss += loss.item()

                    

                    score_val = score_val/len(val_ld)

                    val_loss = val_loss/len(val_ld)             

                    

                    

                print(f'train_loss: {train_loss:.4f}, valid_loss: {val_loss:.4f}, valid_score: {score_val:.4f}')

                schedule.step(val_loss)



                if score_val >= val_score_max:

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
train_process = Trainer(config, base_model)



train_process.train(folds=5,

                    epochs=5,

                    train=train,

                    check_number=5)