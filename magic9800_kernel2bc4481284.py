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
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
train = pd.read_csv('../input/labeledTrainData.tsv',header=0,delimiter="\t",quoting=3)
def review_to_words(raw_review):
    #remove <br></br>
    example1 = BeautifulSoup(raw_review,features='lxml')
    #print(example1.get_text())


    #remove punctuation
    letters_only = re.sub("[^a-zA-Z]"," ",example1.get_text())
    #print(letters_only)

    lower_case = letters_only.lower()
    words = lower_case.split()

    #nltk.download()
    #print(stopwords.words('english'))
    stops = set(stopwords.words('english'))
    #print(stops)
    words = [w for w in words if  w not in stops]
    return " ".join(words)

num_reviews = train['review'].size
clean_train_review = []
for i in range(0,num_reviews):
    if(i %1000 ==0):
        print('Review %d of %d \n ' % (i+1,num_reviews))
    clean_train_review.append(review_to_words(train['review'][i]))

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer='word',preprocessor=None,tokenizer=None,stop_words=None,max_features=5000)
train_data_features = vectorizer.fit_transform(clean_train_review)
#print(type(train_data_features))
train_data_features = train_data_features.toarray()
#print(train_data_features.shape)

#take a look at the words in the vocabulary
vocab = vectorizer.get_feature_names()
#print(vocab)

#the counts of each word in the vocabulary
dist = np.sum(train_data_features,axis=0)
#for tag,count in zip(vocab,dist):
#    print(count,tag)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(train_data_features,train['sentiment'])

#test testSet
test = pd.read_csv('../input/testData.tsv',delimiter='\t',quoting=3,header=0)
clean_test_review = []
for i in range(0,test['review'].size):
    clean_test_review.append(review_to_words(test['review'][i]))
test_data_features = vectorizer.transform(clean_test_review).toarray()

result = rf.predict(test_data_features)

res = pd.DataFrame({'id':test['id'],'sentiment':result})
res.to_csv('result.csv',index=False,quoting=3)