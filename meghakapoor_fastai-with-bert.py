# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)



%reload_ext autoreload

%autoreload 2

%matplotlib inline



from fastai import *

from fastai.text import *

from fastai.tabular import *



from pathlib import Path

from typing import *



import torch

import torch.optim as optim



import gc

gc.collect()



import re

import os

import re

import gc

import pickle  

import random

import keras



import numpy as np

import pandas as pd

import tensorflow as tf

import tensorflow_hub as hub

import keras.backend as K



from keras.models import Model

from keras.layers import Dense, Input, Dropout, Lambda

from keras.optimizers import Adam

from keras.callbacks import Callback

from scipy.stats import spearmanr, rankdata

from os.path import join as path_join

from numpy.random import seed

from urllib.parse import urlparse

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import KFold, train_test_split

from sklearn.linear_model import LogisticRegression

from bayes_opt import BayesianOptimization

from lightgbm import LGBMRegressor

from nltk.tokenize import wordpunct_tokenize

from nltk.stem.snowball import EnglishStemmer

from nltk.stem import WordNetLemmatizer

from functools import lru_cache

from tqdm import tqdm as tqdm

from fastai.text import *

from fastai.metrics import *
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True



SEED = 42

seed_everything(SEED)
train = pd.read_csv("../input/google-quest-challenge/train.csv")

test = pd.read_csv("../input/google-quest-challenge/test.csv")

sub = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")
train.shape, test.shape, sub.shape
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',

 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '\n', '\xa0', '\t',

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

    x = str(x)

    for punct in puncts:

        x = x.replace(punct, f' {punct} ')

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





def clean_data(df, columns: list):

    for col in columns:

        df[col] = df[col].apply(lambda x: clean_numbers(x))

        df[col] = df[col].apply(lambda x: clean_text(x.lower()))

        df[col] = df[col].apply(lambda x: replace_typical_misspell(x))



    return df
import pandas as pd

from bs4 import BeautifulSoup

from nltk.corpus import stopwords

import re

import string
uri_re = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'



def stripTagsAndUris(x):

    if x:

        # BeautifulSoup on content

        soup = BeautifulSoup(x, "html.parser")

        # Stripping all <code> tags with their content if any

        if soup.code:

            soup.code.decompose()

        # Get all the text out of the html

        text =  soup.get_text()

        # Returning text stripping out all uris

        return re.sub(uri_re, "", text)

    else:

        return ""
train["question_title"] = train["question_title"].map(stripTagsAndUris)

train["question_body"] = train["question_body"].map(stripTagsAndUris)

train["answer"] = train["answer"].map(stripTagsAndUris)



test["question_title"] = test["question_title"].map(stripTagsAndUris)

test["question_body"] = test["question_body"].map(stripTagsAndUris)

test["answer"] = test["answer"].map(stripTagsAndUris)
def removePunctuation(x):

    # Lowercasing all words

    x = x.lower()

    # Removing non ASCII chars

    x = re.sub(r'[^\x00-\x7f]',r' ',x)

    # Removing (replacing with empty spaces actually) all the punctuations

    return re.sub("["+string.punctuation+"]", " ", x)
train["question_title"] = train["question_title"].map(removePunctuation)

train["question_body"] = train["question_body"].map(removePunctuation)

train["answer"] = train["answer"].map(removePunctuation)



test["question_title"] = test["question_title"].map(removePunctuation)

test["question_body"] = test["question_body"].map(removePunctuation)

test["answer"] = test["answer"].map(removePunctuation)
stops = set(stopwords.words("english"))

def removeStopwords(x):

    # Removing all the stopwords

    filtered_words = [word for word in x.split() if word not in stops]

    return " ".join(filtered_words)
train["question_title"] = train["question_title"].map(removeStopwords)

train["question_body"] = train["question_body"].map(removeStopwords)

train["answer"] = train["answer"].map(removeStopwords)



test["question_title"] = test["question_title"].map(removeStopwords)

test["question_body"] = test["question_body"].map(removeStopwords)

test["answer"] = test["answer"].map(removeStopwords)
target_cols_questions = ['question_asker_intent_understanding',

       'question_body_critical', 'question_conversational',

       'question_expect_short_answer', 'question_fact_seeking',

       'question_has_commonly_accepted_answer',

       'question_interestingness_others', 'question_interestingness_self',

       'question_multi_intent', 'question_not_really_a_question',

       'question_opinion_seeking', 'question_type_choice',

       'question_type_compare', 'question_type_consequence',

       'question_type_definition', 'question_type_entity',

       'question_type_instructions', 'question_type_procedure',

       'question_type_reason_explanation', 'question_type_spelling',

       'question_well_written']



target_cols_answers = ['answer_helpful',

       'answer_level_of_information', 'answer_plausible', 'answer_relevance',

       'answer_satisfaction', 'answer_type_instructions',

       'answer_type_procedure', 'answer_type_reason_explanation',

       'answer_well_written']



targets = target_cols_questions + target_cols_answers



input_columns = ['question_title', 'question_body', 'answer']
train = clean_data(train, ['answer', 'question_body', 'question_title'])

test = clean_data(test, ['answer', 'question_body', 'question_title'])
train.head()
find = re.compile(r"^[^.]*")



train['netloc_1'] = train['url'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])

test['netloc_1'] = test['url'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])



train['netloc_2'] = train['question_user_page'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])

test['netloc_2'] = test['question_user_page'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])



train['netloc_3'] = train['answer_user_page'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])

test['netloc_3'] = test['answer_user_page'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])
train = train[input_columns + targets]

test = test[input_columns]
train, val = train_test_split(train, test_size=0.2, shuffle=True, random_state=42)
train.shape, val.shape
!pip install ../input/sacremoses/sacremoses-master/

!pip install ../input/transformers/transformers-master/
!ls ../input/pretrained-bert-models-for-pytorch/bert-base-uncased
!ls ../input/distilbertbaseuncased/
from collections import defaultdict

from dataclasses import dataclass

import functools

import gc

import itertools

import json

from multiprocessing import Pool

import os

from pathlib import Path

import random

import re

import shutil

import subprocess

import time

from typing import Callable, Dict, List, Generator, Tuple

from os.path import join as path_join



import numpy as np

import pandas as pd

from pandas.io.json._json import JsonReader

from sklearn.preprocessing import LabelEncoder

from tqdm._tqdm_notebook import tqdm_notebook as tqdm



import torch

from torch import nn, optim

from torch.utils.data import Dataset, Subset, DataLoader



from transformers import BertTokenizer, AdamW, BertModel, BertPreTrainedModel, BertConfig, DistilBertConfig, DistilBertModel, DistilBertTokenizer, DistilBertForSequenceClassification

from transformers.optimization import get_linear_schedule_with_warmup
# # From the Ref Kernel's

# from math import floor, ceil



# def _get_masks(tokens, max_seq_length):

#     """Mask for padding"""

#     if len(tokens)>max_seq_length:

#         raise IndexError("Token length more than max seq length!")

#     return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))



# def _get_segments(tokens, max_seq_length):

#     """Segments: 0 for the first sequence, 1 for the second"""

    

#     if len(tokens) > max_seq_length:

#         raise IndexError("Token length more than max seq length!")

        

#     segments = []

#     first_sep = True

#     current_segment_id = 0

    

#     for token in tokens:

#         segments.append(current_segment_id)

#         if token == "[SEP]":

#             if first_sep:

#                 first_sep = False 

#             else:

#                 current_segment_id = 1

#     return segments + [0] * (max_seq_length - len(tokens))



# def _get_ids(tokens, tokenizer, max_seq_length):

#     """Token ids from Tokenizer vocab"""

    

#     token_ids = tokenizer.convert_tokens_to_ids(tokens)

#     input_ids = token_ids + [0] * (max_seq_length-len(token_ids))

#     return input_ids



# def _trim_input(title, question, answer, max_sequence_length=512, t_max_len=30, q_max_len=239, a_max_len=239):

    

#     #293+239+30 = 508 + 4 = 512

#     t = tokenizer.tokenize(title)

#     q = tokenizer.tokenize(question)

#     a = tokenizer.tokenize(answer)

    

#     t_len = len(t)

#     q_len = len(q)

#     a_len = len(a)



#     if (t_len+q_len+a_len+4) > max_sequence_length:

        

#         if t_max_len > t_len:

#             t_new_len = t_len

#             a_max_len = a_max_len + floor((t_max_len - t_len)/2)

#             q_max_len = q_max_len + ceil((t_max_len - t_len)/2)

#         else:

#             t_new_len = t_max_len

      

#         if a_max_len > a_len:

#             a_new_len = a_len 

#             q_new_len = q_max_len + (a_max_len - a_len)

#         elif q_max_len > q_len:

#             a_new_len = a_max_len + (q_max_len - q_len)

#             q_new_len = q_len

#         else:

#             a_new_len = a_max_len

#             q_new_len = q_max_len

            

            

#         if t_new_len+a_new_len+q_new_len+4 != max_sequence_length:

#             raise ValueError("New sequence length should be %d, but is %d"%(max_sequence_length, (t_new_len + a_new_len + q_new_len + 4)))

        

#         t = t[:t_new_len]

#         q = q[:q_new_len]

#         a = a[:a_new_len]

    

#     return t, q, a



# def _convert_to_bert_inputs(title, question, answer, tokenizer, max_sequence_length):

#     """Converts tokenized input to ids, masks and segments for BERT"""

    

#     stoken = ["[CLS]"] + title + ["[SEP]"] + question + ["[SEP]"] + answer + ["[SEP]"]



#     input_ids = _get_ids(stoken, tokenizer, max_sequence_length)

#     input_masks = _get_masks(stoken, max_sequence_length)

#     input_segments = _get_segments(stoken, max_sequence_length)



#     return [input_ids, input_masks, input_segments]



# # def compute_input_arays(df, columns, tokenizer, max_sequence_length):

    

# #     input_ids, input_masks, input_segments = [], [], []

# #     for _, instance in tqdm(df[columns].iterrows()):

# #         t, q, a = instance.question_title, instance.question_body, instance.answer

# #         t, q, a = _trim_input(t, q, a, max_sequence_length)

# #         ids, masks, segments = _convert_to_bert_inputs(t, q, a, tokenizer, max_sequence_length)

# #         input_ids.append(ids)

# #         input_masks.append(masks)

# #         input_segments.append(segments)

# #     return [

# #         torch.from_numpy(np.asarray(input_ids, dtype=np.int32)).long(), 

# #         torch.from_numpy(np.asarray(input_masks, dtype=np.int32)).long(),

# #         torch.from_numpy(np.asarray(input_segments, dtype=np.int32)).long(),

# #     ]



# # def compute_output_arrays(df, columns):

# #     return np.asarray(df[columns])
# tokenizer = BertTokenizer.from_pretrained("../input/pretrained-bert-models-for-pytorch/bert-base-uncased-vocab.txt")

# input_categories_train = list(train.columns[[0,1,2]])

# input_categories_val = list(val.columns[[0,1,2]])

# input_categories_test = list(test.columns[[0,1,2]])
# input_categories_test
# max_sequence_length = 512

# input_ids_train, input_masks_train, input_segments_train = _convert_to_bert_inputs(train['question_title'], train['question_body'], train['answer'], tokenizer, max_sequence_length)
# %%time

# outputs_train = compute_output_arrays(train, columns = targets)

# outputs_val = compute_output_arrays(val, columns = targets)



# inputs_train = compute_input_arays(train, input_categories_train, tokenizer, max_sequence_length=512)

# inputs_val = compute_input_arays(val, input_categories_val, tokenizer, max_sequence_length=512)



# test_inputs = compute_input_arays(test, input_categories_test, tokenizer, max_sequence_length=512)
# %%time

# lengths_train = np.argmax(inputs_train[0] == 0, axis=1)

# lengths_train[lengths_train == 0] = inputs_train[0].shape[1]

# y_train_torch = torch.tensor(train[targets].values, dtype=torch.float32)



# lengths_val = np.argmax(inputs_val[0] == 0, axis=1)

# lengths_val[lengths_val == 0] = inputs_val[0].shape[1]

# y_val_torch = torch.tensor(val[targets].values, dtype=torch.float32)



# sequences = np.array(test_inputs[0])

# lengths_test = np.argmax(sequences == 0, axis=1)

# lengths_test[lengths_test == 0] = sequences.shape[1]
# y_train_torch
# from torch.utils import data

# dataset_train = data.TensorDataset(inputs_train[0], #input_ids

#                              inputs_train[1], #input_masks

#                              inputs_train[2], #input_segments

#                              y_train_torch, #targets,

#                              lengths_train, #lengths of each seq

#                             )





# dataset_val = data.TensorDataset(inputs_val[0], #input_ids

#                              inputs_val[1], #input_masks

#                              inputs_val[2], #input_segments

#                              y_val_torch, #targets,

#                              lengths_val, #lengths of each seq

#                             )





# dataset_test = data.TensorDataset(test_inputs[0], #input_ids

#                              test_inputs[1], #input_masks

#                              test_inputs[2], #input_segments

#                              torch.from_numpy(lengths_test), #lengths of each seq

#                             )
# next(iter(dataset_train))
# BATCH_SIZE = 32

# train_loader = data.DataLoader(dataset_train,

#                                batch_size=BATCH_SIZE,

#                                shuffle=True,

#                                drop_last=True,

#                               )



# val_loader = data.DataLoader(dataset_val,

#                                batch_size=BATCH_SIZE,

#                                shuffle=True,

#                                drop_last=True,

#                               )



# test_loader = data.DataLoader(dataset_test,

#                                batch_size=BATCH_SIZE,

#                                shuffle=False,

#                                drop_last=False,

#                               )

# next(iter(train_loader)) #input_ids, input_masks, input_segments, targets, lengths
# y_train_torch[0], len(y_train_torch[0])
# Creating a config object to store task specific information

class Config(dict):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        for k, v in kwargs.items():

            setattr(self, k, v)

    

    def set(self, key, val):

        self[key] = val

        setattr(self, key, val)

        

config = Config(

    testing=False,

    seed = 42,

    roberta_model_name='bert-base-uncased', 

    use_fp16=False,

    bs=32, 

#     max_seq_len=512, 

    hidden_dropout_prob=.25,

    hidden_size=768, 

    start_tok = "[CLS]",

    end_tok = "[SEP]",

)
# forward tokenizer



class FastAiRobertaTokenizer_t(BaseTokenizer):

    """Wrapper around RobertaTokenizer to be compatible with fastai"""

    def __init__(self, tokenizer: BertTokenizer, max_seq_len: int=30, **kwargs): 

        self._pretrained_tokenizer = tokenizer

        self.max_seq_len = max_seq_len 

    def __call__(self, *args, **kwargs): 

        return self 

    def tokenizer(self, t:str) -> List[str]: 

        """Adds Roberta bos and eos tokens and limits the maximum sequence length""" 

        return [config.start_tok] + self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2] + [config.end_tok]
# forward tokenizer



class FastAiRobertaTokenizer_q(BaseTokenizer):

    """Wrapper around RobertaTokenizer to be compatible with fastai"""

    def __init__(self, tokenizer: BertTokenizer, max_seq_len: int=120, **kwargs): 

        self._pretrained_tokenizer = tokenizer

        self.max_seq_len = max_seq_len 

    def __call__(self, *args, **kwargs): 

        return self 

    def tokenizer(self, t:str) -> List[str]: 

        """Adds Roberta bos and eos tokens and limits the maximum sequence length""" 

        return [config.start_tok] + self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2] + [config.end_tok]
# forward tokenizer



class FastAiRobertaTokenizer_a(BaseTokenizer):

    """Wrapper around RobertaTokenizer to be compatible with fastai"""

    def __init__(self, tokenizer: BertTokenizer, max_seq_len: int=400, **kwargs): 

        self._pretrained_tokenizer = tokenizer

        self.max_seq_len = max_seq_len 

    def __call__(self, *args, **kwargs): 

        return self 

    def tokenizer(self, t:str) -> List[str]: 

        """Adds Roberta bos and eos tokens and limits the maximum sequence length""" 

        return [config.start_tok] + self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2] + [config.end_tok]
# create fastai tokenizer 

bert_tok = BertTokenizer.from_pretrained('../input/pretrained-bert-models-for-pytorch/bert-base-uncased-vocab.txt')



fastai_tokenizer_t = Tokenizer(tok_func=FastAiRobertaTokenizer_t(bert_tok, max_seq_len=30), 

                             pre_rules=[], post_rules=[])

fastai_tokenizer_q = Tokenizer(tok_func=FastAiRobertaTokenizer_q(bert_tok, max_seq_len=120), 

                             pre_rules=[], post_rules=[])

fastai_tokenizer_a = Tokenizer(tok_func=FastAiRobertaTokenizer_a(bert_tok, max_seq_len=400), 

                             pre_rules=[], post_rules=[])

# create fastai vocabulary 

path = Path()

bert_tok.save_vocabulary(path)

   

fastai_bert_vocab = Vocab(list(bert_tok.vocab.keys()))
train.head()
databunch_1 = TextDataBunch.from_df(".", train, val, test,

                  tokenizer=fastai_tokenizer_t,

                  vocab=fastai_bert_vocab,

                  include_bos=False,

                  include_eos=False,

                  text_cols=input_columns[0],

                  label_cols=targets,

                  bs=32,

                  mark_fields=True,

                  collate_fn=partial(pad_collate, pad_first=False, pad_idx=0),

             )



databunch_2 = TextDataBunch.from_df(".", train, val, test,

                  tokenizer=fastai_tokenizer_q,

                  vocab=fastai_bert_vocab,

                  include_bos=False,

                  include_eos=False,

                  text_cols=input_columns[1],

                  label_cols=targets,

                  bs=8,

                  mark_fields=True,

                  collate_fn=partial(pad_collate, pad_first=False, pad_idx=0),

             )



databunch_3 = TextDataBunch.from_df(".", train, val, test,

                  tokenizer=fastai_tokenizer_a,

                  vocab=fastai_bert_vocab,

                  include_bos=False,

                  include_eos=False,

                  text_cols=input_columns[2],

                  label_cols=targets,

                  bs=8,

                  mark_fields=True,

                  collate_fn=partial(pad_collate, pad_first=False, pad_idx=0),

             )



databunch_1.save('databunch_1.pkl')

databunch_2.save('databunch_2.pkl')

databunch_3.save('databunch_3.pkl')
databunch_1.show_batch()
databunch_2.show_batch()
databunch_3.show_batch()
start_time = time.time()



seed = 42



num_labels = len(targets)

n_epochs = 3

lr = 2e-5

warmup = 0.05

batch_size = 32

accumulation_steps = 4



bert_model_config = '../input/pretrained-bert-models-for-pytorch/bert-base-uncased/bert_config.json'



bert_model = 'bert-base-uncased'

do_lower_case = 'uncased' in bert_model

device = torch.device('cuda')



output_model_file = 'bert_pytorch.bin'

output_optimizer_file = 'bert_pytorch_optimizer.bin'

output_amp_file = 'bert_pytorch_amp.bin'



random.seed(seed)

np.random.seed(seed)

torch.manual_seed(seed)

torch.cuda.manual_seed(seed)

torch.backends.cudnn.deterministic = True
class BertForSequenceClassification(BertPreTrainedModel):

    r"""

        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:

            Labels for computing the sequence classification/regression loss.

            Indices should be in ``[0, ..., config.num_labels - 1]``.

            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),

            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:

        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:

            Classification (or regression if config.num_labels==1) loss.

        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``

            Classification (or regression if config.num_labels==1) scores (before SoftMax).

        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)

            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)

            of shape ``(batch_size, sequence_length, hidden_size)``:

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.

        **attentions**: (`optional`, returned when ``config.output_attentions=True``)

            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1

        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1

        outputs = model(input_ids, labels=labels)

        loss, logits = outputs[:2]

    """

    def __init__(self, config):

        super(BertForSequenceClassification, self).__init__(config)

        self.num_labels = config.num_labels



        self.bert = BertModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)



        self.init_weights()



    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,

                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):



        outputs = self.bert(input_ids,

                            attention_mask=attention_mask,

                            token_type_ids=token_type_ids,

                            position_ids=position_ids,

                            head_mask=head_mask,

                            inputs_embeds=inputs_embeds)



        pooled_output = outputs[1]



        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)



        return logits
loss_func = nn.BCEWithLogitsLoss()
def reduce_loss(loss, reduction='sum'):

    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss



class LabelSmoothingCrossEntropy(nn.Module):

    def __init__(self, ε:float=0.1, reduction='sum'):

        super().__init__()

        self.ε,self.reduction = ε,reduction

    

    def forward(self, output, target):

        c = output.size()[-1]

        log_preds = F.log_softmax(output, dim=-1)

        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)

        nll = F.nll_loss(log_preds, target.long(), reduction=self.reduction)

        return lin_comb(loss/c, nll, self.ε)
# class LabelSmoothingLoss(nn.Module):

#     def __init__(self, classes, smoothing=0.0, dim=-1):

#         super(LabelSmoothingLoss, self).__init__()

#         self.confidence = 1.0 - smoothing

#         self.smoothing = smoothing

#         self.cls = classes

#         self.dim = dim



#     def forward(self, pred, target):

#         pred = pred.log_softmax(dim=self.dim)

#         with torch.no_grad():

#             # true_dist = pred.data.clone()

#             true_dist = torch.zeros_like(pred)

#             true_dist.fill_(self.smoothing / (self.cls - 1))

#             true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

#         return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
loss_func = nn.BCEWithLogitsLoss()
bert_config = BertConfig.from_json_file(bert_model_config)

bert_config.num_labels = len(targets)



model_path = os.path.join('../input/pretrained-bert-models-for-pytorch/bert-base-uncased/')



model = BertForSequenceClassification.from_pretrained(model_path, config=bert_config)



learn_bert_1 = Learner(databunch_1, model, loss_func=loss_func, model_dir='/temp/model')

learn_bert_2 = Learner(databunch_2, model, loss_func=loss_func, model_dir='/temp/model')

learn_bert_3 = Learner(databunch_3, model, loss_func=loss_func, model_dir='/temp/model')
model.bert.embeddings
def bert_clas_split(self) -> List[nn.Module]:

    

    bert = model.bert

    embedder = bert.embeddings

    pooler = bert.pooler

    encoder = bert.encoder

    classifier = [model.dropout, model.classifier]

    n = len(encoder.layer)//3

    print(n)

    groups = [[embedder], list(encoder.layer[:n]), list(encoder.layer[n+1:2*n]), list(encoder.layer[(2*n)+1:]), [pooler], classifier]

    return groups
x = bert_clas_split(model)
learn_bert_1.layer_groups
learn_bert_1.summary()
learn_bert_1.split([x[1],  x[3],  x[5]])
learn_bert_1.layer_groups
learn_bert_1.freeze_to(-1)
learn_bert_1.summary()
learn_bert_1.lr_find()
import seaborn as sns

from matplotlib import pyplot as plt

import matplotlib.style as style

style.use('seaborn-poster')

style.use('ggplot')
learn_bert_1.recorder.plot(suggestion=True)
learn_bert_1.fit_one_cycle(3, max_lr=slice(1e-3, 5e-3), moms=(0.8,0.7), pct_start=0.2, wd =1.)
learn_bert_1.freeze_to(-2)

learn_bert_1.summary()
learn_bert_1.lr_find()

learn_bert_1.recorder.plot(suggestion=True)
learn_bert_1.fit_one_cycle(3, max_lr=slice(1e-5, 1e-3), moms=(0.8,0.7), pct_start=0.4, wd =1.)
learn_bert_1.freeze_to(-3)

learn_bert_1.summary()
learn_bert_1.lr_find()

learn_bert_1.recorder.plot(suggestion=True)
learn_bert_1.fit_one_cycle(3, max_lr=slice(1e-6, 1e-4), moms=(0.8,0.7), pct_start=0.4, wd =1.)
learn_bert_1.unfreeze()

learn_bert_1.summary()
learn_bert_1.lr_find()

learn_bert_1.recorder.plot(suggestion=True)
learn_bert_1.fit_one_cycle(3, slice(1e-7, 1e-5), moms=(0.8,0.7), pct_start=0.4, wd =1.)
def get_ordered_preds(learn_bert_1, ds_type, preds):

    np.random.seed(42)

    sampler = [i for i in learn_bert_1.data.dl(ds_type).sampler]

    reverse_sampler = np.argsort(sampler)

    preds = [p[reverse_sampler] for p in preds]

    return preds
test_raw_preds = learn_bert_1.get_preds(ds_type=DatasetType.Test)

test_preds_bert_1 = get_ordered_preds(learn_bert_1, DatasetType.Test, test_raw_preds)

test_preds_bert_1 = torch.FloatTensor(test_preds_bert_1[0])
test_preds_bert_1
learn_bert_2.split([x[1],  x[3],  x[5]])
learn_bert_2.freeze_to(-1)

learn_bert_2.summary()
learn_bert_2.lr_find()

learn_bert_2.recorder.plot(suggestion=True)
learn_bert_2.fit_one_cycle(10, max_lr=slice(1e-3, 5e-3), moms=(0.8,0.7), pct_start=0.2, wd =1.)
learn_bert_2.freeze_to(-2)

learn_bert_2.summary()
learn_bert_2.lr_find()

learn_bert_2.recorder.plot(suggestion=True)
learn_bert_2.fit_one_cycle(10, max_lr=slice(1e-5, 1e-4), moms=(0.8,0.7), pct_start=0.2, wd =1.)
learn_bert_2.freeze_to(-3)

learn_bert_2.summary()
learn_bert_2.lr_find()

learn_bert_2.recorder.plot(suggestion=True)
learn_bert_2.fit_one_cycle(3, max_lr=slice(1e-5, 1e-4), moms=(0.8,0.7), pct_start=0.2, wd =1.)
learn_bert_2.unfreeze()

learn_bert_2.summary()
learn_bert_2.lr_find()

learn_bert_2.recorder.plot(suggestion=True)
learn_bert_2.fit_one_cycle(2, max_lr=slice(1e-6, 1e-5), moms=(0.8,0.7), pct_start=0.2, wd =1.5)
def get_ordered_preds(learn_bert_2, ds_type, preds):

    np.random.seed(42)

    sampler = [i for i in learn_bert_2.data.dl(ds_type).sampler]

    reverse_sampler = np.argsort(sampler)

    preds = [p[reverse_sampler] for p in preds]

    return preds
test_raw_preds = learn_bert_2.get_preds(ds_type=DatasetType.Test)

test_preds_bert_2 = get_ordered_preds(learn_bert_2, DatasetType.Test, test_raw_preds)

test_preds_bert_2 = torch.FloatTensor(test_preds_bert_2[0])

test_preds_bert_2
learn_bert_3.split([x[1],  x[3],  x[5]])
learn_bert_3.freeze_to(-1)

learn_bert_3.summary()
learn_bert_3.lr_find()

learn_bert_3.recorder.plot(suggestion=True)
learn_bert_3.fit_one_cycle(10, max_lr=slice(1e-3, 5e-3), moms=(0.8,0.7), pct_start=0.2, wd =1.)
learn_bert_3.freeze_to(-2)

learn_bert_3.summary()
learn_bert_3.lr_find()

learn_bert_3.recorder.plot(suggestion=True)
learn_bert_3.fit_one_cycle(10, max_lr=slice(1e-5, 1e-4), moms=(0.8,0.7), pct_start=0.2, wd =1.)
learn_bert_3.freeze_to(-3)

learn_bert_3.summary()
learn_bert_3.lr_find()

learn_bert_3.recorder.plot(suggestion=True)
learn_bert_3.fit_one_cycle(3, max_lr=slice(1e-5, 1e-4), moms=(0.8,0.7), pct_start=0.2, wd =1.5)
learn_bert_3.unfreeze()

learn_bert_3.summary()
learn_bert_3.lr_find()

learn_bert_3.recorder.plot(suggestion=True)
learn_bert_3.fit_one_cycle(2, max_lr=slice(1e-5, 1e-4), moms=(0.8,0.7), pct_start=0.2, wd =1.5)
def get_ordered_preds(learn_bert_3, ds_type, preds):

    np.random.seed(42)

    sampler = [i for i in learn_bert_3.data.dl(ds_type).sampler]

    reverse_sampler = np.argsort(sampler)

    preds = [p[reverse_sampler] for p in preds]

    return preds
test_raw_preds = learn_bert_3.get_preds(ds_type=DatasetType.Test)

test_preds_bert_3 = get_ordered_preds(learn_bert_3, DatasetType.Test, test_raw_preds)

test_preds_bert_3 = torch.FloatTensor(test_preds_bert_3[0])

test_preds_ber_3
final_preds_test = (test_preds_bert_1 + test_preds_bert_2  + test_preds_bert_3 )/3
sub.iloc[:, 1:] = final_preds_test.numpy()

sub.to_csv('submission.csv', index=False)

sub.head()
fig, axes = plt.subplots(6, 5, figsize=(18, 15))

axes = axes.ravel()

bins = np.linspace(0, 1, 20)



for i, col in enumerate(targets):

    ax = axes[i]

    sns.distplot(train[col], label=col, bins=bins, ax=ax, color='blue')

    sns.distplot(sub[col], label=col, bins=bins, ax=ax, color='orange')

    # ax.set_title(col)

    ax.set_xlim([0, 1])

plt.tight_layout()

plt.show()

plt.close()
# y_train = train[targets].values



# for column_ind in range(30):

#     curr_column = y_train[:, column_ind]

#     values = np.unique(curr_column)

#     map_quantiles = []

#     for val in values:

#         occurrence = np.mean(curr_column == val)

#         cummulative = sum(el['occurrence'] for el in map_quantiles)

#         map_quantiles.append({'value': val, 'occurrence': occurrence, 'cummulative': cummulative})

            

#     for quant in map_quantiles:

#         pred_col = final_preds_test[:, column_ind]

#         q1, q2 = np.quantile(pred_col, quant['cummulative']), np.quantile(pred_col, min(quant['cummulative'] + quant['occurrence'], 1))

#         pred_col[(pred_col >= q1) & (pred_col <= q2)] = quant['value']

#         final_preds_test[:, column_ind] = pred_col
# sub.iloc[:, 1:] = final_preds_test.numpy()

# sub.to_csv('submission.csv', index=False)

# sub.head()
# fig, axes = plt.subplots(6, 5, figsize=(18, 15))

# axes = axes.ravel()

# bins = np.linspace(0, 1, 20)



# for i, col in enumerate(targets):

#     ax = axes[i]

#     sns.distplot(train[col], label=col, bins=bins, ax=ax, color='blue')

#     sns.distplot(sub[col], label=col, bins=bins, ax=ax, color='orange')

#     # ax.set_title(col)

#     ax.set_xlim([0, 1])

# plt.tight_layout()

# plt.show()

# plt.close()