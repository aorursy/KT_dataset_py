!pip install contractions -q
#Generic

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re,string,unicodedata

import contractions #import contractions_dict



#FastAI

import fastai

from fastai import *

from fastai.text import * 



#Functional Tool

from functools import partial



#Garbage

import gc



#NLTK

import nltk

from nltk.tokenize.toktok import ToktokTokenizer



#SK Learn libraries

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn import metrics

from sklearn.compose import ColumnTransformer



#Warnings

import warnings

warnings.filterwarnings("ignore")
#Load data

url = '../input/kickstarter-nlp/df_text_eng.csv'

raw_data = pd.read_csv(url, header='infer')
#checking the columns

raw_data.columns
#creating a seperate dataframe

data = raw_data[['blurb','state']]
#inspect the shape of the dataframe

data.shape
#Check for null/missing values in the new dataframe

data.isna().sum()
#Dropping the records with null/missing values

data = data.dropna()
#Checking the records per state

data.groupby('state').size()
#Encoding the State label to convert them to numerical values

label_encoder = LabelEncoder() 



#Applying to the dataset

data['state']= label_encoder.fit_transform(data['state']) 
#inspect the newly created dataframe

data.head()
#Remove special characters & retain alphabets

data['blurb'] = data['blurb'].str.replace("[^a-zA-Z]", " ")
#Lowering the case

data['blurb'] = data['blurb'].str.lower()



#stripping leading spaces (if any)

data['blurb'] = data['blurb'].str.strip()
#removing punctuations

from string import punctuation



def remove_punct(text):

  for punctuations in punctuation:

    text = text.replace(punctuations, '')

  return text



#apply to the dataset

data['blurb'] = data['blurb'].apply(remove_punct)
#function to remove macrons & accented characters

def remove_accented_chars(text):

    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    return text



#applying the function on the clean dataset

data['blurb'] = data['blurb'].apply(remove_accented_chars)
#Function to expand contractions

def expand_contractions(con_text):

  con_text = contractions.fix(con_text)

  return con_text



#applying the function on the clean dataset

data['blurb'] = data['blurb'].apply(expand_contractions) 
#Removing Stopwords

nltk.download('stopwords')



from nltk.corpus import stopwords 

#stop_words = stopwords.words('english')

stopword_list = set(stopwords.words('english'))
#instantiating the tokenizer function

tokenizer = ToktokTokenizer()
#function to remove stopwords

def remove_stopwords(text, is_lower_case=False):

    tokens = tokenizer.tokenize(text)

    tokens = [token.strip() for token in tokens]

    if is_lower_case:

        filtered_tokens = [token for token in tokens if token not in stopword_list]

    else:

        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]

    filtered_text = ' '.join(filtered_tokens)    

    return filtered_text



#applying the function

data['blurb_norm'] = data['blurb'].apply(remove_stopwords) 
#Dropping the "blurb" column

data = data.drop(['blurb'], axis=1)
#Inspect the dataframe after stopword removal

data.head()
#Databack

data_bkup = data.copy()
#data split

train_data, test_data = train_test_split(data, test_size = 0.1, random_state = 12, stratify=data['state'])
train_data.head()
#reseting index for test_data

test_data.reset_index(drop=True, inplace=True)



#resting index for train_data

train_data.reset_index(drop=True, inplace=True)
#Shape of train & test data

print("Training Data Shape - ",train_data.shape, " Test Data Shape - ", test_data.shape)
#Language Model

lang_mod = TextLMDataBunch.from_df(train_df= train_data, valid_df=test_data, path='')



#Classification Model

class_mod = TextClasDataBunch.from_df(path='', train_df=train_data, valid_df=test_data, vocab=lang_mod.train_ds.vocab, bs=32)
lang_learner = language_model_learner(lang_mod, arch = AWD_LSTM, pretrained = True, drop_mult=0.3)
#finding the learning rate for language learner

lang_learner.lr_find()
#Plotting the Recorder Plot

lang_learner.recorder.plot()
#Training the language learner model

lang_learner.fit_one_cycle(1, 1e-2, moms=(0.8,0.7))
#Saving the language learner encoder

lang_learner.save_encoder('fai_langlrn_enc')
class_learner = text_classifier_learner(class_mod, drop_mult=0.3, arch = AWD_LSTM, pretrained = True)

class_learner.load_encoder('fai_langlrn_enc')
#finding the learning rate of this class_learner

class_learner.lr_find()
#Plotting the Recorder Plot for the class learner

class_learner.recorder.plot()
#Training the Class Learner Model

class_learner.fit_one_cycle(1, 1e-3, moms=(0.8,0.7))
#saving the Class Learner Model

class_learner.save_encoder('fai_classlrn_enc_tuned')
#free memory

gc.collect()

class_learner.show_results()
# predictions

pred, trgt = class_learner.get_preds()
#Confusion matrix

prediction = np.argmax(pred, axis = 1)

pd.crosstab (prediction, trgt)
#Prediction on Test Dataset

test_dataset = pd.DataFrame({'blurb': test_data['blurb_norm'], 'actual_state' : test_data['state'] })

test_dataset = pd.concat([test_dataset, pd.DataFrame(prediction, columns = ['predicted_state'])], axis=1)



test_dataset.head()