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
import pandas as pd

import numpy as np

import seaborn as sns

import re

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

stop_words = list(set(stopwords.words('english')))

stop_words.extend(['CA','ca','!','@','#','$','%','&','*','(',')',',','.','?','/','{','}','[',']'])
train = pd.read_csv("../input/nlp-getting-started/train.csv")

test = pd.read_csv("../input/nlp-getting-started/test.csv")
train.head()
train['target'].value_counts()
train['text'][2]
train[train.keyword.notna()]['keyword'].values

# for x in train[train.keyword.notna()]['keyword'].values:

#     print(x.replace('%20',' '))
train['keyword'].isna().sum()
kws= list(train[train.keyword.notna()]['keyword'].values)



for x,y in zip(train.loc[train.keyword.isna(),'text'].head(10),train.loc[train.keyword.isna(),'keyword']):

#     for y in kws:

#         if y in x:

#             print(y)

#             train[train.text==x]['keyword'] = y

    print(x,y)    

    
for x in train.loc[train.keyword.isna(),'text']:

    for y in kws:

        if y.lower() in x.lower():

            print(y)

            train.loc[train.text==x,'keyword'] = y

            break
train['keyword'].isna().sum()
for x in train.loc[train.keyword.isna(),'text']:

    print(x.lower())
for x in train.loc[train.keyword.isna(),'text']:

    if 'heat wave' in x.lower():

        train.loc[train.text==x,'keyword'] = 'heat wave'

    elif 'oil spill' in x.lower():

        train.loc[train.text==x,'keyword'] = 'oil spill'

        
train.keyword.isna().sum()
for x in train.loc[train.keyword.isna(),'text']:

    print(x.lower(),train.loc[train.text==x,'target'])
train['keyword'] = train['keyword'].fillna('general')
train['keyword'].isna().sum()
train.head()
train.loc[train['location'].isna(),'text']
train['location'].isna().sum()
train.shape
train['text'][0][0]
locs = list(train.loc[train.location.notna(),'location'].values)





for x in train.loc[train['location'].isna(),'text']:

    for y in locs:

        if y.lower() in word_tokenize(x.lower()) and y.lower() not in stop_words:

            train.loc[train['text']==x,'location'] = y

            

            break

        

    
train.location.isna().sum()
train.to_csv('modified1.csv')
train.loc[train['location'].isna(),'text']
train['location'] = train['location'].fillna('unknown')
train.location.isna().any()
train.head()
ids = train['id']

train.drop(['id'],axis=1,inplace=True)
train.head()