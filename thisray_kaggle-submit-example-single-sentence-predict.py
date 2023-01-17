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
import pandas as pd

import numpy as np

import json

import os



data_path = "../input"

data_file = 'friends_kaggle.json'

submit_file = 'friends_kaggle_samplesubmit.csv'



out_path = './'

data_file = os.path.join(data_path, data_file)



with open(data_file, 'r') as filehandle:  

    dataList = json.load(filehandle)



print('len(dataList): {}'.format(len(dataList)))



for i in range(3):

    print(dataList[0][i])
from collections import defaultdict



dict_forDF = defaultdict(list)

column_list = list(dataList[0][0].keys())

print('column_list: ', column_list)



df = pd.DataFrame()

for _, dialogue in enumerate(dataList):

    for _, line_dict in enumerate(dialogue):

        for col in column_list:

            dict_forDF[col].append(line_dict[col])

for col in column_list:

    df[col] = dict_forDF[col]

            

df.to_csv(os.path.join(out_path, 'df_friends.csv'), index=False)

df.head(10)
select_emotion = ['joy', 'sadness', 'anger', 'neutral']



df_train = df[df['tag']=='train']

df_train = df_train[df_train['emotion'].isin(select_emotion)]

df_test = df[df['tag']=='test']



x_train = df_train['utterance']

y_train = df_train['emotion']

x_test = df_test['utterance']

y_test = df_test['emotion']  ## Here I give the ground truth directly.



print('x_train.shape: ', x_train.shape)

print('y_train.shape: ', y_train.shape)

print('x_test.shape: ', x_test.shape)

print('y_test.shape: ', y_test.shape)
import nltk

from sklearn.feature_extraction.text import CountVectorizer



BOW = CountVectorizer(max_features=5000, tokenizer=nltk.word_tokenize) 

BOW.fit(x_train)



x_train = BOW.transform(x_train)

x_test = BOW.transform(x_test)



print('x_train.shape: ', x_train.shape)

print('x_test.shape: ', x_test.shape)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

import warnings

warnings.simplefilter("ignore")



model = LogisticRegression()

model.fit(x_train, y_train)



pred_train = model.predict(x_train)

pred_test = model.predict(x_test)



def eval_func(y, pred):

    print('classification_report: \n', classification_report(y, pred))

    print('accuracy_score: ', accuracy_score(y, pred))

    print('confusion_matrix: \n', confusion_matrix(y, pred))



print('train:')

eval_func(y=y_train, pred=pred_train)

print()

print('-----'*12)

print('\ntest:')

eval_func(y=y_test, pred=pred_test)
df_submit = pd.read_csv(os.path.join(data_path, submit_file))

df_submit['emotion'] = pred_test



df_submit.to_csv('./myoutput.csv', index=False)



print(df_submit.head())

print('save ok')
## If you are using kaggle kernel, you could submit the result directly.