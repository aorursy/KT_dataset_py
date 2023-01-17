import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (6,6)



from keras import backend as K

from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints



from keras.layers import Dense, Input, LSTM, Bidirectional, Activation, Conv1D, GRU, TimeDistributed

from keras.layers import Dropout, Embedding, GlobalMaxPooling1D, MaxPooling1D, Add, Flatten, SpatialDropout1D

from keras.layers import GlobalAveragePooling1D, BatchNormalization, concatenate

from keras.layers import Reshape, merge, Concatenate, Lambda, Average

from keras.models import Sequential, Model, load_model

from keras.callbacks import ModelCheckpoint

from keras.initializers import Constant

from keras.layers.merge import add



from keras.preprocessing import sequence

from keras.preprocessing.text import Tokenizer, text_to_word_sequence

from keras.utils import np_utils



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# load data



df = pd.read_json('../input/News_Category_Dataset_v2.json', lines=True)

df['article_id'] = range(1, 1+len(df))

df.head()
cates = df.groupby('category')

print("total categories:", cates.ngroups)

print(cates.size())
import spacy

from spacy import displacy

nlp = spacy.load('en')
doc = df.loc[1,'headline']

doc = nlp(doc)

displacy.render(doc,style='ent',jupyter=True)
from tqdm import tqdm, tqdm_notebook
nlp = spacy.load('en',

                 disable=['parser', 

                          'tagger',

                          'textcat'])
frames = []

for i in tqdm_notebook(range(100)):

    doc = df.loc[i,'headline']

    text_id = df.loc[i,'article_id']

    category = df.loc[i,'category']

    authors = df.loc[i,'authors']

    doc = nlp(doc)

    ents = [(e.text,  e.label_) 

            for e in doc.ents 

            if len(e.text.strip(' -—')) > 0]

    frame = pd.DataFrame(ents)

    frame['article_id'] = text_id

#    frame['category'] = category

#    frame['authors'] = authors

    frames.append(frame)
npf = pd.concat(frames)

npf.columns = ["value", "type","article_id"]

npf.head(10)

npf.set_index(['article_id', 'type', 'value'], append=True)

p = npf.pivot(index='article_id', columns='type', values='value')



print(p)
from collections import OrderedDict

from pandas import DataFrame

import pandas as pd

import numpy as np



table = OrderedDict((

    ("Item", ['Item0', 'Item0', 'Item1', 'Item1']),

    ('CType',['Gold', 'Bronze', 'Gold', 'Silver']),

    ('USD',  ['1$', '2$', '3$', '4$']),

    ('EU',   ['1€', '2€', '3€', '4€'])

))

d = DataFrame(table)

print(type(d))



p = d.pivot(index='Item', columns='CType', values='USD')

#print(p)