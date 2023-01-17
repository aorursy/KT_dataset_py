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
df=pd.read_csv('/kaggle/input/email-spam-classification/emails.csv')
df.shape
df.isnull().sum()
df=df.iloc[:,0:2]
#removing null data
df
df.isnull().sum()
df.shape
#removing remaining null data
df.dropna(subset=['spam'],inplace=True)
df.shape
df.sample(5)
#one means spam 0 means no spm
df.sample()['text'].values
#Naive Bayes --> Predictor
#New Email --> .....
#Spam classifer will read content of email and predict whether its spam or not
#we will need vectorisation or rather text vectorisation(making a table of numerical data from a paragraph)
df.head()
#text preprocessing
#remove stock words and stemming(the, an etc...and for ex sing singing will all become singing)

#1 convert to lowercase.

def toLowerCase(text):
    return text.lower()
df['text'].apply(toLowerCase)
df['text']=df['text'].apply(toLowerCase)
#2 remove special chars

def removeSpecialChar(text):
    x=''
    for i in text:
        if i.isalnum():
            x=x+i
        else:
            x=x+ ' '
    return x
removeSpecialChar('this has been a rainy $ # day')
df['text']=df['text'].apply(removeSpecialChar)
from nltk.corpus import stopwords
stopwords.words('english')
def removeStopWords(text):
    x=[]
    for i in text.split():
        if i not in stopwords.words('english'):
            x.append(i)
    y=x[:]
    
    x.clear()
    
    return y
df['text']=df['text'].apply(removeStopWords)
#stemming 
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
def stemText(text):
    z=[]
    for i in text:
        z.append(ps.stem(i))
    w=z[:]
    z.clear()
    return w
ps.stem('singing')

#this is what stemming does
df['text']=df['text'].apply(stemText)
def joinList(text):
    return ' '.join(text)
df['text']=df['text'].apply(joinList)
df.head()
#text preprocessing done
#now convert to vectors
from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_features=6000)
#max_features is basically a hyperparameter that helps set the number of words you wanna vectorize
cv.fit_transform(df['text']).toarray()
#every single inner array represents one email
cv.fit_transform(df['text']).toarray().shape

#5729 is no. of emails and 29147 is no. of unique words
X=cv.fit_transform(df['text']).toarray()
y=df.iloc[:,-1].values
y
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.naive_bayes import GaussianNB
clf1= GaussianNB()
from sklearn.naive_bayes import MultinomialNB
clf2=MultinomialNB()
from sklearn.naive_bayes import BernoulliNB
clf3=BernoulliNB()

#these are different types of distributions and we are implmenting all 3 to compare results for each and use the best.
#make histogram and get probability from pdf

clf1.fit(X_train,y_train)
clf2.fit(X_train,y_train)
clf3.fit(X_train,y_train)
y_pred1=clf1.predict(X_test)
y_pred2=clf2.predict(X_test)
y_pred3=clf3.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred1,y_test))
print(accuracy_score(y_pred2,y_test))
print(accuracy_score(y_pred3,y_test))
