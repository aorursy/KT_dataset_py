# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv',encoding='latin-1')
df.head()
df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)
df.head()
df.shape
tokens=[]
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords
df1=pd.DataFrame()

df1['v2']=df[df['v1']=='spam']['v2']

df1.reset_index(inplace = True, drop = True) 

for i in range(len(df1)):

    tokens+=word_tokenize(df1['v2'][i])
without_stopwords=[]
from nltk.corpus import stopwords
for token in tokens:

    if token not in stopwords.words('english') and not  token.isnumeric():

        without_stopwords.append(token.lower())
from nltk.probability import FreqDist
bag_of_words=FreqDist()
for i in without_stopwords:

    if i in bag_of_words:

        bag_of_words[i]+=1

    else:

        bag_of_words[i]=1
top_two_thousand=[]
count=0

for key,value in sorted(bag_of_words.items(),key=lambda x: x[1],reverse=True):

    if count==2000:

        break

    #print(key,value)

    top_two_thousand.append(key)

    count+=1
documents=[]

for i in range(len(df)):

    documents.append((df['v2'][i],df['v1'][i]))
def find_features(document):

    words = word_tokenize(document)

    l_words=[x.lower() for x in words]

    features = {}

    for w in top_two_thousand:

        features[w] = (w in l_words)

    return features
featuresets = [(find_features(sms), category) for (sms, category) in documents]
import random
random.shuffle(featuresets)



training_set = featuresets[:4001]

testing_set = featuresets[4001:]
from nltk import NaiveBayesClassifier

classifier = NaiveBayesClassifier.train(training_set)
import nltk

print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)
tests=['I really like it', 

       'I do not think this is good one', 

       'this is good one',

       'I hate the show!',

      'apply today, win a nokia!!',

      'Thank you for having our service',

      'watch latest shows on netflix for free!!']



for test in tests:

    features = {word: (word in word_tokenize(test.lower())) for word in tokens}

    print(test," : ", classifier.classify(features))