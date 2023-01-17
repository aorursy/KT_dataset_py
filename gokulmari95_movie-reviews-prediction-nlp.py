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
import nltk
df=pd.read_csv('../input/labeledTrainData.tsv',delimiter='\t',quoting=3)

df.head(4)
df.info()
df.describe()
df['review'][0]
from bs4 import BeautifulSoup

ex=BeautifulSoup(df['review'][0])

ex
import re

from nltk.corpus import stopwords

def clean_text(review):

    r_text=BeautifulSoup(review).get_text()

    letters=re.sub('[^a-zA-Z]',' ',r_text)

    words=letters.lower().split()

    no_stop=set(stopwords.words('english'))

    meaningful=[w for w in words if w not in no_stop]

    return(' '.join(meaningful))
cr=clean_text(df['review'][4])

print(cr)
clean_review_df=[]

for i in range(0,len(df['review'])):

    clean_review_df.append(clean_text(df['review'][i]))
clean_review_df[1]
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer='word',tokenizer=None,

stop_words=None,max_features=6000)

train_df_feat=vectorizer.fit_transform(clean_review_df)

train_df_feat=train_df_feat.toarray()
train_df_feat.shape
vocab=vectorizer.get_feature_names()

print(vocab)
from sklearn.ensemble import RandomForestClassifier

forest=RandomForestClassifier(n_estimators=100)

forest=forest.fit(train_df_feat,df['sentiment'])
test=pd.read_csv("../input/testData.tsv",

                header=0, delimiter="\t",quoting=3 )

print(test.shape)

num_revw=len(test['review'])
clean_test_review=[]

for i in range(0,num_revw):

    clean_test=clean_text(test['review'][i])

    clean_test_review.append(clean_test)
clean_test_review[0]
test_data_features = vectorizer.transform(clean_test_review)

test_data_features = test_data_features.toarray()



result=forest.predict(test_data_features)
output=pd.DataFrame(data={"id":test['id'],"Review":result})

test['sentiment']=result
test #this sentiment column is predicted one!