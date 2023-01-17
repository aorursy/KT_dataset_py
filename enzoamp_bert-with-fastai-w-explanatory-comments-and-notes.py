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

    '''

    Seeds all sources of randomness in machine

    '''

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
# List of punctuations to handle

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',

 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '\n', '\xa0', '\t',

 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '\u3000', '\u202f',

 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '«',

 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]



# Dictionary to handle misspellings and to standardized how phrases are written

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

    '''

    1. Handles punctuations by adding whitespace to both sides of the punctuation.

    2. This allows the NLP model to account for the "meaning" of punctuations separately from the words themselves.

    3. This also increases the frequency of the tokens (less sparse), which makes them more effective features.

    '''

    x = str(x)

    for punct in puncts:

        x = x.replace(punct, f' {punct} ')

    return x





def clean_numbers(x):

    """

    Replace numbers with as many `#` as there are digits in the number (min 2, max 5).

    This can give numbers more meaning in the context of NLP modelling.

    This is because it now turns into a higher frequency token that indicates, whether it's a big number (i.e. #####) or a small number (i.e. ##)

    """

    x = re.sub('[0-9]{5,}', '#####', x)

    x = re.sub('[0-9]{4}', '####', x)

    x = re.sub('[0-9]{3}', '###', x)

    x = re.sub('[0-9]{2}', '##', x)

    return x





def _get_mispell(mispell_dict):

    """

    Returns the mispelling dictionary and the regex that's supposed to apply all the mappings to the text

    """

    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))

    return mispell_dict, mispell_re





def replace_typical_misspell(text):

    """

    Replace misspellings with the "correct" spellings

    """

    

    mispellings, mispellings_re = _get_mispell(mispell_dict)



    def replace(match):

        return mispellings[match.group(0)]



    return mispellings_re.sub(replace, text)





def clean_data(df, columns: list):

    """

    Apply all the text processing functions above to the text

    """

    for col in columns:

        df[col] = df[col].apply(lambda x: clean_numbers(x))

        df[col] = df[col].apply(lambda x: clean_text(x.lower()))

        df[col] = df[col].apply(lambda x: replace_typical_misspell(x))



    return df
# Distinguish between the target columns and the input columns



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
# Clean data using the preprocessing script

train = clean_data(train, ['answer', 'question_body', 'question_title'])

test = clean_data(test, ['answer', 'question_body', 'question_title'])
# Parse the urls from the question and answer text and extract the domains as features



find = re.compile(r"^[^.]*")



train['netloc_1'] = train['url'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])

test['netloc_1'] = test['url'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])



train['netloc_2'] = train['question_user_page'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])

test['netloc_2'] = test['question_user_page'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])



train['netloc_3'] = train['answer_user_page'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])

test['netloc_3'] = test['answer_user_page'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])
train['netloc_1'].head(), train['netloc_2'].head(), train['netloc_3'].head()
# Filter to input and target columns for the training set

train = train[input_columns + targets]



# Filter to only input columns for the test set

test = test[input_columns]
# Split dataset to a train and validation set with a 20% random sample for the test set

train, val = train_test_split(train, test_size=0.2, shuffle=True, random_state=42)
train.shape, val.shape
# Installing packages from local

!pip install ../input/sacremoses/sacremoses-master/

!pip install ../input/transformers/transformers-master/
!ls ../input/pretrained-bert-models-for-pytorch/bert-base-uncased
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



from transformers import BertTokenizer, AdamW, BertModel, BertPreTrainedModel, BertConfig

from transformers.optimization import get_linear_schedule_with_warmup
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

    roberta_model_name='bert-base-uncased', # can also be exchnaged with roberta-large 

    use_fp16=False,

    bs=16, 

    max_seq_len=128, 

    hidden_dropout_prob=.25,

    hidden_size=768, # 1024 for roberta-large

    start_tok = "[CLS]",

    end_tok = "[SEP]",

)
# forward tokenizer



class FastAiRobertaTokenizer(BaseTokenizer):

    """Wrapper around RobertaTokenizer to be compatible with fastai"""

    def __init__(self, tokenizer: BertTokenizer, max_seq_len: int=128, **kwargs): 

        self._pretrained_tokenizer = tokenizer

        self.max_seq_len = max_seq_len 

    def __call__(self, *args, **kwargs): 

        return self 

    def tokenizer(self, t:str) -> List[str]: 

        """Adds Roberta bos and eos tokens and limits the maximum sequence length""" 

        return [config.start_tok] + self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2] + [config.end_tok]
# backward tokenizer



class FastAiRobertaTokenizerBackward(BaseTokenizer):

    """Wrapper around RobertaTokenizer to be compatible with fastai"""

    def __init__(self, tokenizer: BertTokenizer, max_seq_len: int=128, **kwargs): 

        self._pretrained_tokenizer = tokenizer

        self.max_seq_len = max_seq_len 

    def __call__(self, *args, **kwargs): 

        return self 

    def tokenizer(self, t:str) -> List[str]: 

        """Adds Roberta bos and eos tokens and limits the maximum sequence length""" 

        return [config.end_tok] + self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2] + [config.start_tok]
# create fastai tokenizer for roberta

bert_tok = BertTokenizer.from_pretrained('../input/pretrained-bert-models-for-pytorch/bert-base-uncased-vocab.txt')



# Create fastai tokenizer from bert tokenizer

fastai_tokenizer = Tokenizer(tok_func=FastAiRobertaTokenizer(bert_tok, max_seq_len=config.max_seq_len), 

                             pre_rules=[], post_rules=[])



# Create fastai backward tokenizer from bert tokenizer

fastai_tokenizer_bwd = Tokenizer(tok_func=FastAiRobertaTokenizerBackward(bert_tok, max_seq_len=config.max_seq_len), 

                             pre_rules=[], post_rules=[])
# create fastai vocabulary for roberta

path = Path()

bert_tok.save_vocabulary(path)

   

fastai_bert_vocab = Vocab(list(bert_tok.vocab.keys()))
# Create fastai databunch



databunch = TextDataBunch.from_df(".", train, val, test,

                  tokenizer=fastai_tokenizer,

                  vocab=fastai_bert_vocab,

                  include_bos=False,

                  include_eos=False,

                  text_cols=input_columns,

                  label_cols=targets,

                  bs=16,

                  mark_fields=True,

                  collate_fn=partial(pad_collate, pad_first=False, pad_idx=0),

             )



# Save databunch file

databunch.save('databunch.pkl')
# Load the databunch from a saved file



databunch = load_data(path, 'databunch.pkl', bs=16)
# Show one batch of data



databunch.show_batch()
# Note that the question and answer are concatenated into the same string without separating them with a [SEP] in between

# A proper approach should treat this as a two sequence input (1. Question, 2. Answer). This is the formatting used for SQuAD



databunch.single_ds[0]
start_time = time.time()



seed = 42



num_labels = len(targets)

n_epochs = 3

lr = 2e-5

warmup = 0.05

batch_size = 16

accumulation_steps = 4



bert_model_config = '../input/pretrained-bert-models-for-pytorch/bert-base-uncased/bert_config.json'



# Uncased version of BERT is used - this means capitalizations aren't accounted for by the model

bert_model = 'bert-base-uncased'

do_lower_case = 'uncased' in bert_model

device = torch.device('cuda')



# Output files from model training

output_model_file = 'bert_pytorch.bin'

output_optimizer_file = 'bert_pytorch_optimizer.bin'

output_amp_file = 'bert_pytorch_amp.bin'



# Setting seeds to make "randomness" reproducible

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
bert_config = BertConfig.from_json_file(bert_model_config)

bert_config.num_labels = len(targets)



model_path = os.path.join('../input/pretrained-bert-models-for-pytorch/bert-base-uncased/')



model = BertForSequenceClassification.from_pretrained(model_path, config=bert_config)

learn_bert = Learner(databunch, model, loss_func=loss_func, model_dir='/temp/model')
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
learn_bert.layer_groups
learn_bert
learn_bert.split([x[2],  x[4],  x[5]])
learn_bert.freeze()
learn_bert.lr_find()
import seaborn as sns

from matplotlib import pyplot as plt

import matplotlib.style as style

style.use('seaborn-poster')

style.use('ggplot')
learn_bert.recorder.plot(suggestion=True)
learn_bert.fit_one_cycle(7, max_lr=slice(1e-3, 1e-2), moms=(0.8,0.7), pct_start=0.2, wd =0.1)
learn_bert.save('head-1')
learn_bert.freeze_to(-2)

learn_bert.fit_one_cycle(7, max_lr=slice(1e-4, 1e-3), moms=(0.8,0.7), pct_start=0.4, wd =0.1)
learn_bert.save('head-2')
learn_bert.freeze_to(-3)

learn_bert.fit_one_cycle(7, max_lr=slice(1e-5, 1e-4), moms=(0.8,0.7), pct_start=0.3, wd =0.1)
learn_bert.unfreeze()

learn_bert.lr_find()

learn_bert.recorder.plot(suggestion=True)
learn_bert.fit_one_cycle(12, slice(1e-5, 1e-4), moms=(0.8,0.7), pct_start=0.4, wd =0.1)
bs, bptt = 32, 80



data_lm = TextLMDataBunch.from_df('.', train, val, test,

                  include_bos=False,

                  include_eos=False,

                  text_cols=['question_title', 'question_body', 'answer'],

                  label_cols=targets,

                  bs=bs,

                  mark_fields=True,

                  collate_fn=partial(pad_collate, pad_first=False, pad_idx=0),

             )



data_lm.save('data_lm.pkl')
path = "."

data_lm = load_data(path, 'data_lm.pkl', bs=bs, bptt=bptt)
path = "."

data_bwd = load_data(path, 'data_lm.pkl', bs=bs, bptt = bptt, backwards=True)
data_lm.show_batch()
data_bwd.show_batch()
awd_lstm_lm_config = dict( emb_sz=400, n_hid=1150, n_layers=3, pad_token=1, qrnn=False, bidir=False, output_p=0.1,

                          hidden_p=0.15, input_p=0.25, embed_p=0.02, weight_p=0.2, tie_weights=True, out_bias=True)
awd_lstm_clas_config = dict(emb_sz=400, n_hid=1150, n_layers=3, pad_token=1, qrnn=False, bidir=False, output_p=0.4,

                       hidden_p=0.2, input_p=0.6, embed_p=0.1, weight_p=0.5)
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5,

                               config=awd_lstm_lm_config, pretrained = False)

learn = learn.to_fp16(clip=0.1)
fnames = ['../input/awd-lstm/lstm_wt103.pth','../input/awd-lstm/itos_wt103.pkl']

learn.load_pretrained(*fnames, strict=False)

learn.freeze()
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(2, max_lr=slice(5e-3, 5e-2), moms=(0.8, 0.7), pct_start=0.3, wd =0.1)
learn.save('fit_head')
learn.unfreeze()

learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(10, max_lr = slice(1e-4, 1e-3), moms=(0.8, 0.7), pct_start=0.3, wd =0.1)
learn.recorder.plot_losses()
learn.save('fine-tuned')

learn.load('fine-tuned')

learn.save_encoder('fine-tuned-fwd')
learn = language_model_learner(data_bwd, AWD_LSTM, drop_mult=0.5,

                               config=awd_lstm_lm_config, pretrained = False)

learn = learn.to_fp16(clip=0.1)
fnames = ['../input/awd-lstm/lstm_wt103.pth','../input/awd-lstm/itos_wt103.pkl']

learn.load_pretrained(*fnames, strict=False)

learn.freeze()
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(2, max_lr=slice(5e-2, 1e-1), moms=(0.8, 0.7), pct_start=0.3, wd =0.1)
learn.save('fit_head-bwd')
learn.unfreeze()

learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(10, max_lr = slice(1e-4, 1e-3), moms=(0.8, 0.7), pct_start=0.3, wd =0.1)
learn.recorder.plot_losses()
learn.save('fine-tuned-bwd')

learn.load('fine-tuned-bwd')

learn.save_encoder('fine-tuned-bwd')
text_cols = ['question_title', "question_body", 'answer']
data_cls = TextClasDataBunch.from_df('.', train, val, test, vocab = data_lm.vocab,

                  include_bos=False,

                  include_eos=False,

                  text_cols=text_cols,

                  label_cols=targets,

                  bs=bs,

                  mark_fields=True,

                  collate_fn=partial(pad_collate, pad_first=False, pad_idx=0),

             )



data_cls.save('data_cls.pkl')
data_cls = load_data(path, 'data_cls.pkl', bs=bs)
data_cls.show_batch()
data_cls_bwd = load_data(path, 'data_cls.pkl', bs=bs, backwards=True)
data_cls_bwd.show_batch()
learn = text_classifier_learner(data_cls, AWD_LSTM, drop_mult=0.5,config=awd_lstm_clas_config, pretrained = False, loss_func=loss_func)

learn.load_encoder('fine-tuned-fwd')

learn = learn.to_fp16(clip=0.1)

#learn.loss_func = L1LossFlat()

fnames = ['../input/awd-lstm/lstm_wt103.pth','../input/awd-lstm/itos_wt103.pkl']

learn.load_pretrained(*fnames, strict=False)

learn.freeze()
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(2, max_lr=slice(1e-2, 1e-1), moms=(0.8, 0.7), pct_start=0.3, wd =0.1)
learn.save('first-head')

learn.load('first-head')
learn.freeze_to(-2)

learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7), pct_start=0.3, wd =0.1)
learn.save('second')

learn.load('second')
learn.freeze_to(-3)

learn.fit_one_cycle(2, slice(1e-4/(2.6**4),1e-4), moms=(0.8,0.7), pct_start=0.3, wd =0.1)
learn.save('third')

learn.load('third')
learn.unfreeze()

learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(7, slice(1e-4/(2.6**4),1e-4), moms=(0.8,0.7), pct_start=0.3, wd =0.1)
learn.recorder.plot_losses()
learn.save('fwd-cls')
learn_bwd = text_classifier_learner(data_cls_bwd, AWD_LSTM, drop_mult=0.5, config=awd_lstm_clas_config, loss_func=loss_func,

                                    pretrained = False)

learn_bwd.load_encoder('fine-tuned-bwd')

learn_bwd = learn_bwd.to_fp16(clip=0.1)
fnames = ['../input/awd-lstm/lstm_wt103.pth','../input/awd-lstm/itos_wt103.pkl']

learn_bwd.load_pretrained(*fnames, strict=False)

learn_bwd.freeze()
learn_bwd.lr_find()

learn_bwd.recorder.plot(suggestion=True)
learn_bwd.fit_one_cycle(2, max_lr=slice(5e-2, 1e-1), moms=(0.8, 0.7), pct_start=0.3, wd =0.1)
learn_bwd.save('first-head-bwd')

learn_bwd.load('first-head-bwd')
learn_bwd.freeze_to(-2)

learn_bwd.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7), pct_start=0.3, wd =0.1)
learn_bwd.save('second-bwd')

learn_bwd.load('second-bwd')
learn_bwd.freeze_to(-3)

learn_bwd.fit_one_cycle(2, slice(1e-5/(2.6**4),1e-5), moms=(0.8,0.7), pct_start=0.3, wd =0.1)
learn_bwd.save('third-bwd')

learn_bwd.load('third-bwd')
learn_bwd.unfreeze()

learn_bwd.lr_find()

learn_bwd.recorder.plot(suggestion=True)
learn_bwd.fit_one_cycle(7, slice(1e-5/(2.6**4),1e-5), moms=(0.8,0.7), pct_start=0.3, wd =0.1)
learn_bwd.recorder.plot_losses()
learn_bwd.save('bwd-cls')
def get_ordered_preds(learn_bert, ds_type, preds):

  np.random.seed(42)

  sampler = [i for i in learn_bert.data.dl(ds_type).sampler]

  reverse_sampler = np.argsort(sampler)

  preds = [p[reverse_sampler] for p in preds]

  return preds
test_raw_preds = learn_bert.get_preds(ds_type=DatasetType.Test)

test_preds_bert = get_ordered_preds(learn_bert, DatasetType.Test, test_raw_preds)
pred_fwd_test, lbl_fwd_test = learn.get_preds(ds_type=DatasetType.Test,ordered=True)

pred_bwd_test, lbl_bwd_test = learn_bwd.get_preds(ds_type=DatasetType.Test,ordered=True)
type(pred_fwd_test)
test_preds_bert = torch.FloatTensor(test_preds_bert[0])
final_preds_test = (0.3*pred_fwd_test + 0.3*pred_bwd_test + 0.4*test_preds_bert)
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

#         pred_col = test_preds_bert[0][:, column_ind]

#         q1, q2 = np.quantile(pred_col, quant['cummulative']), np.quantile(pred_col, min(quant['cummulative'] + quant['occurrence'], 1))

#         pred_col[(pred_col >= q1) & (pred_col <= q2)] = quant['value']

#         test_preds_bert[0][:, column_ind] = pred_col
# sub.iloc[:, 1:] = test_preds_bert[0].numpy()

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