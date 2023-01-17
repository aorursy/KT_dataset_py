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
data=pd.read_csv("../input/train.csv")

data.iloc[22]['question_text'] #iloc is information in that location
from wordcloud import WordCloud
import matplotlib.pyplot as plt

wc= WordCloud().generate('i love india, i have its culture')
plt.imshow(wc)
#when we have multiple lines, we join and make it into one string 
x=['a','c','d','c','e']
' '.join(x)
questions_string=' '.join(data['question_text'])#here we are combining all the lines into a single string
wc=WordCloud().generate(questions_string) 
plt.imshow(wc)

insincere_questions=data[data['target']==1]
wc=WordCloud().generate(' '.join(insincere_questions['question_text']))
plt.imshow(wc)
#1.Convert all characters to lower case
docs=data['question_text'].str.lower()

#2.Apply regular expressions to retain only alphabets
docs= docs.str.replace('[^a-z ]','') #except alphabets everything is replaced with space
docs.head()

#3. Remove commonly used words
#which we will find through nltk library where 250 words are listed as commonly used words
#for which we will import nltk library

import nltk
stopwords=nltk.corpus.stopwords.words('english')
stopwords


len(stopwords)# length of stopwords



#creating a user defined function
#def remove_stopwords(text):
#    words=nltk.word_tokenize(text)
#    print(words)
#    print('-------')

# split sentence into words
#go word by word using loop to check if it exist in stopwords, remove it else keep it
#def remove_stopwords(text):
#    words=nltk.word_tokenize(text)
#    words=[word for word in words if word not in stopwords]
#    print(words)
#    print('-------')

def remove_stopwords(text):
    words=nltk.word_tokenize(text)
    words=[stemmer.stem(word) for word in words if word not in stopwords]
    #print(words)
    #print('-------')
    return' '.join(words)
#docs.head(2).apply(remove_stopwords)
docs_clean=docs.apply(remove_stopwords)
docs_clean.head()

#cresting stemmer
#nltk has lot of stemmer in which porterstemmer is widely used
stemmer= nltk.stem.PorterStemmer()
stemmer.stem('plays')
#but sometimes it change the meaning as well for example organisation to orgaN, we have to use it samrtly
stemmer.stem('organisation')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

vectorizer= CountVectorizer()
train, validate= train_test_split(docs_clean, test_size=0.3,random_state=100)
vectorizer=CountVectorizer()
vectorizer.fit(train)
train_dtm=vectorizer.transform(train)
validate_dtm=vectorizer.transform(validate)
train.shape

train_dtm #here we get Compressed Sparse Row format,914285 is number of rows in training dataset,
          #143417 is number of distinct words
          #which is created as column

#here 5628198 only contains values out of 914285x143417, rest of them contains only 0s
percentage_of_non_zero_values= 5628198 / (914285*143417)*100

percentage_of_non_zero_values #which is less than 1 percent
#pd.DataFrame(train_dtm[:5].toarray()) #here we took only 1st 5 row
pd.DataFrame(train_dtm[:5].toarray(), columns=vectorizer.get_feature_names())

train_x=train_dtm
validate_x=validate_dtm
train_y=data.loc[train.index]['target']
validate_y=data.loc[validate.index]['target']




from sklearn.ensemble import RandomForestClassifier
model_rf= RandomForestClassifier(n_estimators=300, random_state=100)
model_rf.fit(train_x,train_y)

validate_pred_class=model_rf.predict(validate_x)