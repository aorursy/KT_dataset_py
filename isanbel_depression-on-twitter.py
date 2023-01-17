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
data = pd.read_csv("/kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv",encoding='latin-1')

data.head()
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "TweetText"]

data.columns = DATASET_COLUMNS
data.describe(include='all')
data.dtypes
import copy

data_ = copy.deepcopy(data)



positif_data = data_[data_.target==4].iloc[:80000,:]

negative_data = data_[data_.target==0].iloc[:80000,:]



sub_data = pd.concat([positif_data,negative_data],axis=0)
data.boxplot(column='target')
data_target=data.groupby('target')
data['target'].value_counts()
data.head()
data_ = {'target': data['target'], 'date': data['date']}

df = pd.DataFrame(data_)

df.head()
# lets ensure the 'date' column is in date format

df['date'] = pd.to_datetime(df['date'])
hour = [ df['date'][i].hour for i in range(len(df['date'])) ]

df['hour'] = hour

df.head()
hour_data = {'0': [0]*24, '2': [0]*24, '4': [0]*24}

for i in range(len(df['hour'])):

    target = str(df['target'][i])

    hour = int(df['hour'][i])

    hour_data[target][hour] += 1
hour_data = [hour_data['0'], hour_data['2'], hour_data['4']]

# Transpose

hour_data = list(map(list,zip(*hour_data)))
df1 = pd.DataFrame(hour_data,index = [i for i in range(24)],columns=['negative', 'neutral', 'positive'])
df1.plot()
positive_at_count = 0

negative_at_count = 0

TweetTextList = list(sub_data['TweetText'])

targetList = list(sub_data['target'])

for i in range(len(sub_data['TweetText'])):

    if TweetTextList[i].find('@') != -1:

        if targetList[i] == 4:

            positive_at_count += 1

        else:

            negative_at_count += 1

at_counts = [positive_at_count, negative_at_count]
import matplotlib.pyplot as plt

names = ['positive', 'negative']

values = [positive_at_count, negative_at_count]

plt.bar(names, values)
import copy

newdata = copy.deepcopy(sub_data)

newdata.drop(['ids','date','flag','user'],axis = 1,inplace = True)
import wordcloud

from wordcloud import WordCloud

import matplotlib.pyplot as plt

%matplotlib inline



import re

import nltk

import string

import warnings



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os
all_words = ' '.join([text for text in newdata['TweetText']])

wordcloud = WordCloud(width=800,height=500,random_state=21,max_font_size=110).generate(all_words)

plt.figure(figsize=(10,7))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.show()
positive_words = ' '.join([text for text in data['TweetText'][data['target']==4]])

wordcloud = WordCloud(width=800,height=500,random_state=21,max_font_size=110).generate(positive_words)

plt.figure(figsize=(10,7))

plt.imshow(wordcloud,interpolation="bilinear")

plt.axis('off')

plt.show()
negative_words = ' '.join([text for text in data['TweetText'][data['target']==0]])

wordcloud = WordCloud(width=800,height=500,random_state=21,max_font_size=110).generate(negative_words)

plt.figure(figsize=(10,7))

plt.imshow(wordcloud,interpolation="bilinear")

plt.axis('off')

plt.show()
positif_data = data[data.target==4].iloc[:10000,:]

print(positif_data.shape)

negative_data = data[data.target==0].iloc[:10000,:]

print(negative_data.shape)

data = pd.concat([positif_data,negative_data],axis = 0)

print(data.shape)

data.head()
# Removing Twitter Handles (@user)

data['Clean_TweetText'] = data['TweetText'].str.replace("@", "") 

# Removing links

data['Clean_TweetText'] = data['Clean_TweetText'].str.replace(r"http\S+", "") 

# Removing Punctuations, Numbers, and Special Characters

data['Clean_TweetText'] = data['Clean_TweetText'].str.replace("[^a-zA-Z]", " ") 

# Remove stop words

import nltk

stopwords=nltk.corpus.stopwords.words('english')

def remove_stopwords(text):

    clean_text=' '.join([word for word in text.split() if word not in stopwords])

    return clean_text

data['Clean_TweetText'] = data['Clean_TweetText'].apply(lambda text : remove_stopwords(text.lower()))

data.head()
# Text Tokenization and Normalization

data['Clean_TweetText'] = data['Clean_TweetText'].apply(lambda x: nltk.word_tokenize(x))

data.head()
# Now let’s stitch these tokens back together

data['Clean_TweetText'] = data['Clean_TweetText'].apply(lambda x: ' '.join([w for w in x]))

# Removing small words

data['Clean_TweetText'] = data['Clean_TweetText'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

data.head()
from xgboost import XGBClassifier

import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier 

from sklearn.linear_model import LogisticRegression 

from sklearn import svm

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer(stop_words='english') 

cv = count_vectorizer.fit_transform(data['Clean_TweetText'])

cv.shape
X_train,X_test,y_train,y_test = train_test_split(cv,data['target'] , test_size=.2,stratify=data['target'], random_state=42)
# XGBC

xgbc = XGBClassifier(max_depth=6, n_estimators=1000, nthread= 3)

xgbc.fit(X_train,y_train)

prediction_xgb = xgbc.predict(X_test)

print(accuracy_score(prediction_xgb,y_test))
# RandomForest

rf = RandomForestClassifier(n_estimators=1000, random_state=42)

rf.fit(X_train,y_train)

prediction_rf = rf.predict(X_test)

print(accuracy_score(prediction_rf,y_test))