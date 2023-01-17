# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#lets not touch the upper tab 

#lets import from herre lol

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import string



%matplotlib inline
fakenews = pd.read_csv("../input/fake-and-real-news-dataset/Fake.csv")[['title','subject']]

fakenews['type']="fake"



fakenews.tail(5)
realnews = pd.read_csv("../input/fake-and-real-news-dataset/True.csv")[['title','subject']]

realnews['type']='true'

realnews.tail(5)
#lets join two data

news=pd.concat([realnews,fakenews])

news['subject'] = news['subject'].str.lower()

news.index=range(0,len(news['title']))



print(news['type'].value_counts())

news.tail(6)
print("Length of TITLE + NEWS text")

(news['title']).apply(len).describe()
news['subject'].hist(by=news['type'],figsize=(15,5))
#lets make a tokner to convert woerd to tokens

def tokner(word):

    nopunc = "".join([char for char in word if char not in string.punctuation])

    #lets not use porterstemmer for the sake of memory

    #and i am leaving the word "US" as it represents country which will eventually to 'us' and 

    #our model may interpret it as english grammer

    token  = [ word.lower() for word in nopunc.split(" ") if word.lower() not in stopwords.words("english") or word is "US"]

    return (" ".join(token))



#cleaned text must be processed out

tokner("Us is doing coronciruc check all over again")

#lets create a token column

#clearly ML is a memory hogggg

#and my model cannot run in my 

#algorithm here 



#as a result i am only applying on title 

news['tokens']=news['title'].apply(tokner)

news.tail(5)
tfidf = TfidfVectorizer()
vectorizer = tfidf.fit_transform(news["tokens"])

tok = tfidf.transform(["Donald Trum"])

tok.toarray()
# so for classification let's compare uisng my favs

# LogisticRegression

# svm.SVC



from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression
#enters the hero

#train_test_split



from sklearn.model_selection import train_test_split



#feature must be the text in vector

X = vectorizer





#labels

y = news['type']



Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5,random_state=77)

print(Xtrain.shape)

print(ytrain.shape)

svcModel = SVC(C=100)

svcModel.fit(Xtrain,ytrain)

def predict(title):

    ctitle=title

    tokens = tokner(title)

    vect = tfidf.transform([tokens])

    pred = svcModel.predict(vect.toarray())

    print(title)

    print("=> ",pred[0])

    return pred

ytest.shape
type(ytest)
Xtest.toarray()
ypred=svcModel.predict(Xtest)





from sklearn import metrics



accuracy = metrics.accuracy_score(ytest,ypred)

print('The accuracy is  ',accuracy)



print(metrics.classification_report(ytest,ypred))
print("\t\t\t\t\t SOME PREFICTIONS")

print("="*80)



print("\n")



predict("Donald Trump sent to moon after feeling sick")

print("\n")



predict("gaida is rhino")

print('\n')



predict("Real Madrid will win the laliga season 20/21")

print('\n')



predict("Mark Zuckberg is an alien on planet earth")

print('\n')



predict("DARK is a theme not a show")

print('\n')





predict("Liverpool F.C. will be nothing without Virgil van Dijk")

print('\n')
