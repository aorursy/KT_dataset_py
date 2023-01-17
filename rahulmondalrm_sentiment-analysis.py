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
df=pd.read_csv('/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')
df.head()
# One review
df['review'][0]
# Text Cleaning
# 1. Sample 10000 rows
# 2. Remove html tags
# 3. Remove special character
# 4. Converting every thing to lower case
# 5. Removing stop words
# 6. Stemming
df=df.sample(10000)
df.shape
df.info()
df['sentiment'].replace({'positive':1,'negative':0},inplace=True)
df.head()
# re='regular expression' library to remove all HTML tags
import re
clean=re.compile('<.*?>')
re.sub(clean, '',df.iloc[2].review)
# Function to clean html tags
def clean_html(text):
    clean = re.compile('<,*?>')
    return re.sub(clean, '',text)
df['review']=df['review'].apply(clean_html)
# Converting everything to lower

def convert_lower(text):
    return text.lower()
df['review']=df['review'].apply(convert_lower)
# Function to remove special chracter

def remove_special(text):
    x=''
    
    for i in text:
        if i.isalnum():
            x=x+i
        else:
            x=x+''
    return x
df['review']=df['review'].apply(remove_special)
# Remove the stop words
import nltk
from nltk.corpus import stopwords
# List of all the stopwords
stopwords.words('english')
len(stopwords.words('english'))
df
def remove_stopwords(text):
    x=[]
    for i in text.split():
        
        if i not in stopwords.words('english'):
            x.append(i)
    y=x[:]
    x.clear()
    return y
df['review']=df['review'].apply(remove_stopwords)
df
# Perform steming (play or played or playing are all same words)

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
y=[]
def stem_words(text):
    for i in text:
        y.append(ps.stem(i))
    z=y[:]
    y.clear()
    return z
stem_words(['I','loved','loving','it'])
df['review']=df['review'].apply(stem_words)
df
# Join back

def join_back(list_input):
    return ' '.join(list_input)
df['review']=df['review'].apply(join_back)
df['review']
x=df.iloc[:,0:1].values
x.shape
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x=cv.fit_transform(df['review']).toarray()
x.shape
x[0].mean()
y=df.iloc[:,-1].values
y
y.shape
x.shape
# x,y
# Training set
# Test set  (Already know the result)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
x_train.shape
x_test.shape
y_train.shape
y_test.shape
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
clf1=GaussianNB
clf3=MultinomialNB
clf3=BernoulliNB
clf1.fit(x_train,y_train)
clf2.fit(x_train,y_train)
clf3.fit(x_train,y_train)
