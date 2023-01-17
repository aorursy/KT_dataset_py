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
import fastai

from fastai import *

from fastai.text import *

from functools import partial

import io

import seaborn as sns



# READ THE TRAINING DATASET

train_data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

train_data.head()
train_data.info()
train_data.isnull()

sns.heatmap(train_data.isnull(), cmap='viridis')



# Many unknowns in the location column. 

# For the moment, we are not going to use it

train_data = train_data.drop(columns = ['id','location'])

train_data
# REPEATED KEYWORDS

print(f"Total keywords in the training set: {train_data.keyword.nunique()}")

train_data.keyword.value_counts().head(15)
# PERCENTAGE OF REAL DISASTERS WITH EACH KEYWORD

word_probability = train_data.groupby('keyword').agg({'text':np.size, 'target':np.mean}).rename(columns={'text':'counts', 'target':'disaster_probability'}).sort_values('disaster_probability').tail(15)

word_probability
sns.countplot(x='target', data=train_data)
# SOURCE: https://medium.com/datadriveninvestor/identifying-disaster-related-tweets-using-deep-learning-and-natural-language-processing-with-fast-e0dfb790b57a

# GETTING THE DATA READY FOR MODELLING

# When working with text, we change the raw text to a list of words or tokens: TOKENIZATION

# Then, transform those tokens into numbers: NUMERICALIZATION

# The numbers are passed to embedding layers that will convert them into arrays of floats

# These arrays are fed to the model

train_data = train_data.drop(columns = ['keyword'])
# WE DON'T FOLLOW THE PREVIOUS LINK

# DATA PREPROCESSING

# Retain only alphabets

train_data['text'] = train_data['text'].str.replace("[^a-zA-Z]", " ")

train_data
# Get rid of stopwords

from nltk.corpus import stopwords 

stop_words = stopwords.words('english')



#Tokenization

tokenized_doc = train_data['text'].apply(lambda x: x.split())

tokenized_doc
# Remove stop words

tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])

tokenized_doc
# De-tokenization

detokenized_doc = []

for i in range(len(train_data)):

    t = ' '.join(tokenized_doc[i])

    detokenized_doc.append(t)

    

train_data['text'] = detokenized_doc

train_data
# SPLIT THE DATASET INTO TRAINING(60%) AND VALIDATION(40%) SETS

from sklearn.model_selection import train_test_split

train_data, valid_data = train_test_split(train_data, test_size=0.4, random_state=12)
# PREPARE THE DATA FOR THE LANGUAGE MODEL USING TORCH AND FASTAI LIBRARIES

#data_lm = TextLMDataBunch.from_df(train_df=train_data, valid_df=valid_data, path="") # It gives errors if there are NaN values

# BE CAREFUL! WE HAVE TO SPLIT TRAIN_DATA INTO TRAIN_DATA AND VALID_DATA

# WE CAN NOT USE TEST_DATA AS VALID_DATA BECAUSE WE DON'T HAVE COLUMN TARGET

#data_lm



data_lm = (TextList.from_df(df=train_data, cols=['text', 'target']).split_by_rand_pct(0.3).label_for_lm().databunch(bs=256, bptt=80, num_workers=0))

data_lm
# DATA FOR THE CLASSIFIER MODEL DATA

#data_clas = TextClasDataBunch.from_df(path="", train_df = train_data, valid_df = valid_data, vocab = data_lm.train_ds.vocab, bs=32)

#PROBLEMS HERE

data_clas = TextClasDataBunch.from_df(".", train_df=train_data,valid_df=valid_data, vocab=data_lm.train_ds.vocab, text_cols='text', label_cols='target',bs=16)

data_clas

# LEARNER OBJECT

learn = language_model_learner(data_lm, arch = AWD_LSTM, pretrained = True, drop_mult=0.7)
# Train the learner object with learning rate 1e-2

learn.fit_one_cycle(1, 1e-2)

# Save this encoder to use it for classification later

learn.save_encoder('ft_enc')
# Data_clas object to build a classifier with our fine-Tuned encoder

learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.7)

learn.load_encoder('ft_enc')
data_clas.show_batch()
# Fit again the model

learn.fit_one_cycle(1, 1e-2)

learn.freeze_to(-2)

learn.fit_one_cycle(1, slice(5e-3/2., 5e-3))

learn.unfreeze()

learn.fit_one_cycle(1, slice(2e-3/100, 2e-3))

learn.recorder.plot()
# Predictions for the validation set

preds, targets = learn.get_preds() 

predictions = np.argmax(preds, axis = 1)
pd.crosstab(predictions, targets) # It is working with 78% accuracy
test_data = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

test_data.head()

test_data = test_data.drop(columns = ['keyword','location'])
# Predictions for the test set



test_data['target'] = test_data['text'].apply(lambda row: str(learn.predict(row)[0]))

test_data
test_data = test_data.drop('text', axis=1)

test_data
test_data.to_csv('submission.csv', index=False)