# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

fake=pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')

fake.head()

true=pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')

true.head()

true['flag']=1

fake['flag']=0

total_dataset=pd.concat([true,fake],axis=0)

total_dataset

total_dataset=total_dataset.sample(44898)

total_dataset

total_dataset['full_text']=total_dataset['title']+" "+total_dataset['text']

total_dataset['full_text']

df=total_dataset[['full_text','flag']]

df

train_df=df[1:29999]

test_df=df[30000:44908]

import re

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()

#Data Preprocessing

def preprocessing(text):

    text=text.lower()

    text=re.sub('[^a-zA-Z0-9\s]','',text)

    text=text.split()

    text=[ps.stem(word) for word in text if not word in set(stopwords.words('english'))]

    reviews=' '.join(text)

    return reviews

train_df['clean_text']=train_df['full_text'].apply(lambda x:preprocessing(x))

train_df['clean_text']

test_df['clean_text']=test_df['full_text'].apply(lambda x:preprocessing(x))

#Convert to TF_IDF measures

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf=TfidfVectorizer(analyzer='word',max_features=300,norm='l1')

tfidf_train=tfidf.fit_transform(train_df['clean_text'])

tfidf_train

tfidf_test=tfidf.fit_transform(test_df['clean_text'])

test_df=test_df.drop('flag',axis=1)

Y=train_df['flag']

#WordCloud

train_df['full_text']=train_df['full_text'].astype(str)

from wordcloud import WordCloud

import matplotlib.pyplot as plt

text = " ".join(str(each) for each in train_df.full_text.unique())    

wc=WordCloud().generate(text)

plt.imshow(wc)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(tfidf_train,Y,test_size=0.2,random_state=10)

from sklearn.ensemble import RandomForestClassifier

from scipy.stats import randint as sp_randint

rf=RandomForestClassifier(n_estimators=150,n_jobs=-1)

model=rf.fit(x_train,y_train)

pred=model.predict(x_test)

pred

from sklearn.metrics import accuracy_score,f1_score,recall_score

accuracy_score(y_test,pred)

f1_score(y_test,pred)

recall_score(y_test,pred)

pred1=rf.predict(tfidf_test)

submission=pd.DataFrame({'Text':test_df['full_text'],'flag':pred1})

submission[submission['flag']==0]

submission.head()



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

os.listdir('../input')

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session