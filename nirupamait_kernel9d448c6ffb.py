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
%matplotlib inline

import os, sys

import re

import string

import pathlib

import random

from collections import Counter, OrderedDict

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import spacy

from tqdm import tqdm, tqdm_notebook, tnrange

tqdm.pandas(desc='Progress')



import torch

import torch.nn as nn

import torch.optim as optim

from torch.autograd import Variable

import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



import torchtext

from torchtext import data

from torchtext import vocab



from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from sklearn.metrics import accuracy_score



from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity='all'



import warnings

warnings.filterwarnings('ignore')



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from google.colab import drive

drive.mount('/content/drive')
import io





file= "/content/drive/My Drive/Colab Notebooks/train.csv"

df = pd.read_csv(file)

df.drop(['date', 'user'], axis=1)

df.head(5)
fig = plt.figure(figsize=(8,5))

ax = sns.barplot(x=df.target.unique(),y=df.target.value_counts());

ax.set(xlabel='Labels');
df['text'] = df.text.progress_apply(lambda x: re.sub('\n', ' ', x))
def split_train_test(df, test_size=0.2):

    train, val = train_test_split(df, test_size=test_size,random_state=42)

    return train.reset_index(drop=True), val.reset_index(drop=True)
traindf, valdf = split_train_test(df, test_size=0.2)
traindf.shape

traindf.head()

traindf.text.value_counts()
valdf.shape

valdf.head()

valdf.text.value_counts()
traindf.to_csv('traindf.csv', index=False)#

valdf.to_csv('valdf.csv', index=False)
len(traindf)


nlp = spacy.load('en',disable=['parser', 'tagger', 'ner'])

def tokenizer(s): return [w.text.lower() for w in nlp(tweet_clean(s))]
def tweet_clean(text):

    text = re.sub(r'[^A-Za-z0-9]+', ' ', text) # remove non alphanumeric character

    text = re.sub(r'https?:/\/\S+', ' ', text) # remove links

    return text.strip()
txt_field = data.Field(sequential=True, tokenize=tokenizer, include_lengths=True, use_vocab=True)

label_field = data.Field(sequential=False, use_vocab=False, pad_token=None, unk_token=None)



train_val_fields = [

    ('Id', None),

    ('target', label_field),

    ('user', None),

    ('text', txt_field)

]
df1=pd.read_csv('traindf.csv')

len(df1)
#%%time

trainds, valds = data.TabularDataset.splits(path='', format='csv', train='traindf.csv',

                                            validation='valdf.csv', fields=train_val_fields, skip_header=True)
len(trainds)
len(valds)
ex = trainds[0]

type(ex)

trainds.fields.items()

ex.target

ex.text

ex = valds[0]

type(ex)

ex.target

ex.text
!wget http://nlp.stanford.edu/data/glove.twitter.27B.zip
!unzip glove*.zip
!ls
!pwd
%%time

vec = vocab.Vectors('glove.twitter.27B.100d.txt', './content')
%%time

txt_field.build_vocab(trainds, valds, max_size=100000, vectors=vec)

label_field.build_vocab(trainds)
txt_field.vocab.vectors.shape
txt_field.vocab.vectors[txt_field.vocab.stoi['the']]
traindl, valdl = data.BucketIterator.splits(datasets=(trainds, valds), 

                                            batch_sizes=(3,3), 

                                            sort_key=lambda x: len(x.text), 

                                            device=None, 

                                            sort_within_batch=True, 

                                            repeat=False)
len(traindl), len(valdl)
batch = next(iter(traindl))

#type(batch)