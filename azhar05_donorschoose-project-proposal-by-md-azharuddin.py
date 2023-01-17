# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk("/kaggle/input"):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import warnings

warnings.filterwarnings("ignore")



from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.metrics import roc_curve, confusion_matrix



from nltk.corpus import stopwords

from tqdm import tqdm



import nltk

import string

import re

import os

import pickle

from collections import Counter

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from sklearn.preprocessing import StandardScaler,MinMaxScaler

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder

from scipy.sparse import hstack
# loading datasets

train = pd.read_csv("../input/donorschoose-application-screening/train.zip")

test = pd.read_csv("../input/donorschoose-application-screening/test.zip")

resources = pd.read_csv("../input/donorschoose-application-screening/resources.zip")

train = train.sort_values(by="project_submitted_datetime")
train.head()
data=pd.concat([train,test],axis=0,ignore_index=True)
data.isna().any()
# showing percentage of missing values for each column in train dataset



percent_missing = train.isnull().sum() * 100 / len(train)

missing_value_df = pd.DataFrame({'column_name': train.columns,

                                 'percent_missing': percent_missing})

missing_value_df
# showing percentage of missing values for each column in test data

percent_missing = test.isnull().sum() * 100 / len(test)

missing_value_df = pd.DataFrame({'column_name': test.columns,

                                 'percent_missing': percent_missing})

missing_value_df
# droping the columns which have more than 50% missing values

nan_cols=missing_value_df[missing_value_df['percent_missing']>50]['column_name']



data.drop(columns=nan_cols,inplace=True)
data.head()
# handling and preprocessing the resource dataset

resources['priceAll'] = resources['quantity']*resources['price']

resource2= resources.groupby('id').agg({'description':'count',

                            'quantity':'sum',

                            'price':'sum',

                            'priceAll':'sum'}).rename(columns={'description':'items'})

resource2['avgPrice'] = resource2.priceAll / resource2.quantity

numFeatures = ['items', 'quantity', 'price', 'priceAll', 'avgPrice']



for i in ['min', 'max', 'mean']:

    resource2 = resource2.join(resources.groupby('id').agg({'quantity':i,

                                          'price':i,

                                          'priceAll':i}).rename(

                                columns={'quantity':i+'Quantity',

                                         'price':i+'Price',

                                         'priceAll':i+'PriceAll'}).fillna(0))

    numFeatures += [i+'Quantity', i+'Price', i+'PriceAll']



resource2 = resource2.join(resources.groupby('id').agg(

    {'description':lambda x:' '.join(x.values.astype(str))}).rename(

    columns={'description':'resource_description'}))



# Concaneting the preprocessed resource with data

data=data.join(resource2, on='id')
data.head()
# spliting the times which then can be categorized 

data['year'] = data.project_submitted_datetime.apply(lambda x: int(x.split("-")[0]))

data['month'] = data.project_submitted_datetime.apply(lambda x: int(x.split("-")[1]))

data['day']=data.project_submitted_datetime.apply(lambda x: int(x.split("-")[2].split(' ')[0]))

# Label encoding categorical features



encoded_year=LabelEncoder().fit_transform(data['year'])

encoded_month=LabelEncoder().fit_transform(data['month'])

encoded_day=LabelEncoder().fit_transform(data['day'])



train_encoded_year=encoded_year[:len(train)]

train_encoded_year=train_encoded_year.reshape(train_encoded_year.shape[0],1)

train_encoded_month=encoded_month[:len(train)]

train_encoded_month=train_encoded_month.reshape(train_encoded_month.shape[0],1)

train_encoded_day=encoded_day[:len(train)]

train_encoded_day=train_encoded_day.reshape(train_encoded_day.shape[0],1)



test_encoded_year=encoded_year[len(train):]

test_encoded_year=test_encoded_year.reshape(test_encoded_year.shape[0],1)

test_encoded_month=encoded_month[len(train):]

test_encoded_month=test_encoded_month.reshape(test_encoded_month.shape[0],1)

test_encoded_day=encoded_day[len(train):]

test_encoded_day=test_encoded_day.reshape(test_encoded_day.shape[0],1)

# Encoding categorical Features

def label_categorize(data, Col):

    vectorizer = CountVectorizer(binary=True,

                                 ngram_range=(1,1),

                                 tokenizer=lambda x:[a.strip() for a in x.split(',')])

    return vectorizer.fit_transform(data[Col].fillna(''))



tp = label_categorize(data, 'teacher_prefix')

ss = label_categorize(data, 'school_state')

pgc = label_categorize(data, 'project_grade_category')

psc = label_categorize(data, 'project_subject_categories')

pssc = label_categorize(data, 'project_subject_subcategories')

# scalling numerical features



train_numeric_std=StandardScaler().fit_transform(data[:len(train)][numFeatures].fillna(0))

test_numeric_std=StandardScaler().fit_transform(data[len(train):][numFeatures].fillna(0))
# Tokenizing ,cleaning and lemmatizing textual columns

from nltk.corpus import stopwords

from nltk.stem.wordnet import WordNetLemmatizer

L = WordNetLemmatizer()



ps1=data['project_essay_1'].apply(lambda x:' '.join(L.lemmatize(token.lower()) for token in nltk.word_tokenize(x) if token.lower() not in stopwords.words('english')))

ps2=tt=data['project_essay_2'].apply(lambda x:' '.join(L.lemmatize(token.lower()) for token in nltk.word_tokenize(x) if token.lower() not in stopwords.words('english')))

pt=data['project_title'].apply(lambda x:' '.join(L.lemmatize(token.lower()) for token in nltk.word_tokenize(x) if token.lower() not in stopwords.words('english')))

prs=data['project_resource_summary'].apply(lambda x:' '.join(L.lemmatize(token.lower()) for token in nltk.word_tokenize(x) if token.lower() not in stopwords.words('english')))

rs=data['resource_description'].apply(lambda x:' '.join(L.lemmatize(token.lower()) for token in nltk.word_tokenize(x) if token.lower() not in stopwords.words('english')))
# Vectorizing textual columns



def vectorize_txt(data, max_features=10010, ngrams=(1,2), verbose=True):

    vectorizer = CountVectorizer(stop_words='english',

                                max_features=max_features,

                                 binary=True,

                                 ngram_range=ngrams)

    X = vectorizer.fit_transform(data)

    return X, vectorizer.get_feature_names()



vec_ps1,_ = vectorize_txt(data['project_essay_1'], max_features=5000)

vec_ps2,_ = vectorize_txt(data['project_essay_2'], max_features=6000)

vec_prs,_= vectorize_txt(data['project_resource_summary'], max_features=4000)

vec_rd,_= vectorize_txt(data['resource_description'], max_features=4000, ngrams=(1,3))

vec_pt,_= vectorize_txt(data['project_title'], max_features=2000)
# I have done all the data preprocessing steps 

# So,now stacking all the features to be feed intothe model



train_stacked_data=hstack((vec_ps1[:len(train)], vec_ps2[:len(train)], vec_prs[:len(train)], vec_rd[:len(train)], vec_pt[:len(train)],tp[:len(train)], ss[:len(train)], pgc[:len(train)], psc[:len(train)], pssc[:len(train)],train_numeric_std,train_encoded_year,train_encoded_month,train_encoded_day)).tocsr()

test_stacked_data=hstack((vec_ps1[len(train):], vec_ps2[len(train):], vec_prs[len(train):], vec_rd[len(train):], vec_pt[len(train):],tp[len(train):], ss[len(train):], pgc[len(train):], psc[len(train):], pssc[len(train):],test_numeric_std,test_encoded_year,test_encoded_month,test_encoded_day)).tocsr()
# Model buidling for binaryclassification

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

tag_classes=1

model = Sequential()

model.add(Dense(20, input_shape=(train_stacked_data.shape[1],), activation="relu"))

model.add(Dense(10, activation="relu"))

model.add(Dense(tag_classes, activation="sigmoid"))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model fitting 

model.fit(train_stacked_data,np.array(train['project_is_approved']),epochs=10)
# predicting on test data

predicted=model.predict_classes(test_stacked_data)
# Submitting predictions

submission=pd.DataFrame()

submission['id'] = test.id

submission['project_is_approved'] = predicted

submission.to_csv('submission.csv', index=False)