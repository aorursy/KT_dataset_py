# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.tokenize import RegexpTokenizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_train= pd.read_csv('../input/train.tsv',delimiter='\t',encoding='utf-8')
df_test= pd.read_csv('../input/test.tsv',delimiter='\t',encoding='utf-8')

print(df_train.head(1))
print(df_train.columns)
#print(df_train['Phrase'])
#print(df_train)
x_train=df_train['Phrase'].str[1:]  # what is the use of this str thing ?
x_test1=df_test['Phrase'].str[1:]  # what is the use of this str thing ?
phrase=df_test['PhraseId']
#print(phrase)
# remove non- integeres characters from the phrases!
#end=len(df_train['Phrase'])

y_train=df_train['Sentiment']

from nltk.corpus import stopwords
from string import punctuation
import re
stop_words = stopwords.words('english') + list(punctuation)+[1,2,3,4,5,6,7,8,9,0]
#print(stop_words)

# Building the tokenizers first
tokenizer =RegexpTokenizer(r'\w+')    # splitiing the words from sentenecse
def tokenizefunc(text):
    words = tokenizer.tokenize(text.lower())
    #lets create a wordlist of tokens 
    wordlist=[word for word in words if word not in stop_words and len(word)>2]
    wrd=[word for word in wordlist if re.match(r"\D",word,re.I)]   # to ignore numbers 
   # print(wrd)
    return wrd
# to test teh functionalities !
o=tokenizefunc("Hi can you plz tell me the status of the following invoices 6789 567tyui 10th?  ']\]")
print(o)         

x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,test_size=0.0,random_state=0)
#print(x_train.shape)
#print(x_test.shape)
#print(y_train.shape)
#print(y_test.shape)
cv=TfidfVectorizer(stop_words=stop_words,
                          tokenizer=tokenizefunc)
x_trained=cv.fit_transform(x_train)
x=cv.get_feature_names()

#print(x)
mnb=MultinomialNB()
mcl=mnb.fit(x_trained,y_train)

x_tested=cv.transform(x_test1)
pred = mnb.predict(x_tested)
print(pred)

# putting in excelfile
print(type(phrase))
print(type(pred))

df_new=pd.DataFrame()
df_new['PhraseId']=phrase
df_new['Sentiment']=pred
#print(df_new)
# Create a Pandas Excel writer using XlsxWriter as the engine.
#writer = pd.ExcelWriter('submission.csv', engine='csvwriter')
df_new.to_csv('Submissions2.csv',index = False)
#print(df_new)











