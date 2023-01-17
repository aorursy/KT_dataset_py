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
sample=False
from fastai import *

from fastai.text import *
defaults.device = torch.device('cuda',0) if torch.cuda.is_available() else torch. device('cpu')

defaults.device
import os

os.listdir('../input/')
DATA_PATH = Path('/kaggle/input/usinlppracticum/')

DATA_PATH.ls()
lm_data=pd.read_csv(DATA_PATH/'imdb_train.csv')

lm_data.head()
lm_data1=pd.read_csv(DATA_PATH/'imdb_train.csv')

lm_data1['sentiment']=0

lm_data2=pd.read_csv(DATA_PATH/'imdb_test.csv')

lm_data2['sentiment']=0

lm_data= pd.concat([lm_data1, lm_data2], ignore_index=True)

lm_data=lm_data[['review','sentiment']]

lm_data.to_csv('lm_data.csv',index=False)

lm_data.shape
if sample:

    lm_data=pd.read_csv('lm_data.csv').sample(10000).reset_index(drop=True)

else:

    lm_data=pd.read_csv('lm_data.csv')

#------------

lm_data.head()
from sklearn.model_selection import train_test_split



train_lm, val_lm = train_test_split(lm_data,test_size=0.10)

train_lm.shape,val_lm.shape
data_lm = TextLMDataBunch.from_df(DATA_PATH, train_lm,val_lm,text_cols='review', label_cols='sentiment')

data_lm.save('/kaggle/working/data_lm_export.pkl')
AWD_LSTM
learn_lm = language_model_learner(data_lm, AWD_LSTM)
import pickle

wiki_itos = pickle.load(open('/kaggle/input/wikivocab2/itos_wt103.pkl', 'rb'))
vocab = data_lm.vocab
awd = learn_lm.model[0]

print(awd)
enc = learn_lm.model[0].encoder
i, unks = 0, []

while len(unks) < 50:

    if data_lm.vocab.itos[i] not in wiki_itos: unks.append((i,data_lm.vocab.itos[i]))

    i += 1
wiki_words = set(wiki_itos)

imdb_words = set(vocab.itos)
wiki_not_imbdb = wiki_words.difference(imdb_words)

imdb_not_wiki = imdb_words.difference(wiki_words)
wiki_not_imdb_list = []



for i in range(100):

    word = wiki_not_imbdb.pop()

    wiki_not_imdb_list.append(word)

    wiki_not_imbdb.add(word)
imdb_not_wiki_list = []



for i in range(100):

    word = imdb_not_wiki.pop()

    imdb_not_wiki_list.append(word)

    imdb_not_wiki.add(word)
learn_lm.fit_one_cycle(3, 2e-2, moms=(0.8,0.7), wd=0.1)
learn_lm.unfreeze()

learn_lm.fit_one_cycle(20, 2e-3, moms=(0.8,0.7), wd=0.1)
learn_lm.path = Path('/kaggle/working') 

learn_lm.model_dir= Path('.')
learn_lm.save_encoder('fine_tuned_enc')
learn_lm.predict("This was a great movie!")
if sample:

    data_cls=pd.read_csv(DATA_PATH/'imdb_train.csv').sample(1000).reset_index(drop=True)

else:

    data_cls=pd.read_csv(DATA_PATH/'imdb_train.csv')

#----------

data_cls.head()
# Classifier model data

from sklearn.model_selection import train_test_split

train, val = train_test_split(data_cls,test_size=0.001, random_state=42)

label_col= 'sentiment'



label_mapping= {'negative':0,'positive':1}

train[label_col]=train[label_col].map(label_mapping)

val[label_col]=val[label_col].map(label_mapping)

train.head()
data_clas = TextDataBunch.from_df(DATA_PATH, train, val,

                  vocab=data_lm.train_ds.vocab,

                  text_cols="review",

                  label_cols='sentiment',

                  bs=64,device = defaults.device)
learn_c = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.3) #.to_fp16()

learn_c.path = Path('/kaggle/working') 

learn_c.model_dir= Path('.')

learn_c.load_encoder('fine_tuned_enc')

learn_c.freeze()
learn_c.fit_one_cycle(2, 2e-2, moms=(0.8,0.7))
learn_c.freeze_to(-2)

learn_c.fit_one_cycle(2, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))
learn_c.freeze_to(-3)

learn_c.fit_one_cycle(3, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))
learn_c.unfreeze()

learn_c.fit_one_cycle(5, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))
learn_c.predict("This was a bad movie!")
data_test=pd.read_csv(DATA_PATH/'imdb_test.csv')
data_test['sentiment_pred'] = 0
from tqdm import tqdm

sentiment_pred = []

for i in tqdm(range(0, len(data_test))):

    sentiment_pred.append(learn_c.predict(data_test['review'][i])[0])

data_test['sentiment_pred'] = sentiment_pred
data_test.head()
data_test.to_csv('/kaggle/working/testData.csv')
train, val = train_test_split(data_cls,test_size=0.001, random_state=42)

label_col= 'sentiment'



label_mapping= {'negative':0,'positive':1}

train[label_col]=train[label_col].map(label_mapping)

val[label_col]=val[label_col].map(label_mapping)



data_test=pd.read_csv(DATA_PATH/'imdb_test.csv')

data_test['sentiment'] = 0



data_clas = TextDataBunch.from_df(DATA_PATH, data_cls, data_test, vocab=data_lm.train_ds.vocab,  text_cols="review", label_cols='sentiment', bs=64,device = defaults.device)



learn_c = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.3) #.to_fp16()

learn_c.path = Path('/kaggle/working') 

learn_c.model_dir= Path('.')

learn_c.load_encoder('fine_tuned_enc')

learn_c.freeze()



learn_c.fit_one_cycle(2, 2e-2, moms=(0.8,0.7))

learn_c.freeze_to(-2)

learn_c.fit_one_cycle(2, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))

learn_c.freeze_to(-3)

learn_c.fit_one_cycle(3, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))

learn_c.unfreeze()

learn_c.fit_one_cycle(5, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))

#learn_c.predict("This was a bad movie!")



y_test = learn_c.get_preds(data_clas.test_ds)

y_test = y_test[0].argmax(dim=1)

df_test = df_test.assign(prediction=pd.Series(y_test[1]))

df_test.to_csv('/kaggle/working/testDataModern.csv')
y_test = learn_c.get_preds(data_clas.test_ds)

y_test = y_test[0].argmax(dim=1)

df_test = df_test.assign(prediction=pd.Series(y_test[1]))

df_test.to_csv('/kaggle/working/testDataModern.csv')