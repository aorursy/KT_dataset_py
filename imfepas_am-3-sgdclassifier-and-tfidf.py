# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')

df_valid = pd.read_csv('../input/valid.csv')

df_sample_submission = pd.read_csv('../input/sample_submission.csv')
df_train.head()
df_valid.head()
df_sample_submission.head()
df_train.drop(['article_link'], axis=1)

df_valid.drop(['article_link'], axis=1)
from sklearn.feature_extraction.text import TfidfVectorizer 
TfidfVec = TfidfVectorizer()
all_headlines = pd.DataFrame()

all_headlines = pd.concat([df_train, df_valid])
Tfidf_vectorized_data = TfidfVec.fit_transform(all_headlines.headline)
df_train.shape
df_train_vec = Tfidf_vectorized_data[:18696]

df_valid_vec = Tfidf_vectorized_data[18696:]
y_train = df_train.is_sarcastic
from  sklearn.linear_model  import  SGDClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score
df_train_1, df_train_2 = train_test_split(df_train_vec, test_size=0.1)

df_y_1, df_y_2 = train_test_split(y_train, test_size=0.1)
model = SGDClassifier(n_jobs=-1, loss='hinge', random_state=42)
#score com os mesmos dados

print("Teste SGDClassifier")

model.fit(df_train_1, df_y_1)

y_pred = model.predict(df_train_1)



print("accuracy self:")

print(model.score(df_train_1,df_y_1))

print('')



print("roc_auc_self:")

print(roc_auc_score(df_y_1,y_pred))

print('')



#cross validate

model.fit(df_train_1, df_y_1)

y_pred = model.predict(df_train_2)



print("accuracy fit with train 1")

print(model.score(df_train_2,df_y_2))

print('')



print("roc_auc fit with train 1")

print(roc_auc_score(df_y_2,y_pred))

print('')



#cross validate

model.fit(df_train_2, df_y_2)

y_pred = model.predict(df_train_1)



print("accuracy fit with train 2")

print(model.score(df_train_1,df_y_1))

print('')



print("roc_auc fit with train 2")

print(roc_auc_score(df_y_1,y_pred))

print('')

print('')
#score com os mesmos dados completos

print("Teste SGDClassifier")

model.fit(df_train_vec, y_train)

y_pred = model.predict(df_train_vec)



print("accuracy self:")

print(model.score(df_train_vec,y_train))

print('')



print("roc_auc_self:")

print(roc_auc_score(y_train,y_pred))

print('')
pred = model.predict(df_valid_vec)
my_submission=pd.DataFrame({'ID': df_valid['ID'], 'is_sarcastic': pred})

my_submission.to_csv('fepas_submission_3.csv',index=False)