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

for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd

data = pd.read_csv("../input/word2vec-nlp-tutorial/labeledTrainData.tsv", header=0, \

 delimiter="\t", quoting=3)

display(data.head())

print(data.shape)

print(data.columns)
print (data["review"][3])
from bs4 import BeautifulSoup 

example1 = BeautifulSoup(data["review"][3]) 

print (example1.get_text())
import re

letters_only=re.sub("[^a-zA-Z]"," ",example1.get_text())

print (letters_only)
lower_case = letters_only.lower()

words = lower_case.split()

clean_text=(" ".join(words))

print(clean_text)
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

print (stopwords.words("english") )

words = [w for w in words if not w in (stopwords.words("english"))]

print(words)
def review_to_words( raw_review ):

    review_text=BeautifulSoup(raw_review).get_text()

    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 

    words = letters_only.lower().split()

    stops = set(stopwords.words("english"))                  

    meaningful_words = [w for w in words if not w in stops] 

    return( " ".join( meaningful_words )) 
num_reviews = data["review"].size



clean_reviews = []

for i in range( 0, num_reviews ):

 clean_reviews.append( review_to_words(data["review"][i] ) )

print(num_reviews)
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer = "word", \

tokenizer = None, \

 preprocessor = None, \

 stop_words = None, \

 max_features = 5000) 

train_data_features = vectorizer.fit_transform(clean_reviews)

train_data_features = train_data_features.toarray()

print(train_data_features.shape)
X = train_data_features

y = data.sentiment 

'''from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X)

X=scaler.transform(X)'''
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=130)
'''from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

grid={"C":[0.1,0.3,1,3,10]}

LR=LogisticRegression()

LRgrid=GridSearchCV(LR,grid,cv=10)

LRgrid.fit(X_train,y_train)



print("Лучшие параметры: ) ",LR.best_params_)

print("Лучший результат: :",LR.best_score_)'''
from sklearn.linear_model import LogisticRegression

a=[0.1,0.3,1,3,10]

for i in range (5):

    LR = LogisticRegression(C=a[i])

    LR.fit(X_train, y_train)

    print("Точность тренировки: C=",a[i], LR.score(X_test, y_test))
LR = LogisticRegression(C=0.1)

LR.fit(X_train, y_train)

print("Точность тренировки: C=0.1", LR.score(X_test, y_test))
from sklearn.naive_bayes import GaussianNB

NB = GaussianNB()

NB.fit(X_train,y_train)

print("Точность тренировки: ", NB.score(X_test, y_test))

NB.fit(X_test, y_test)

print("Точность тренировки(валидационные): ", NB.score(X_test, y_test))
#Загрузка данных

test = pd.read_csv("../input/word2vec-nlp-tutorial/testData.tsv", header=0, delimiter="\t", \

 quoting=3 )



num_reviews = len(test["review"])

clean_test_reviews = [] 

for i in range(0,num_reviews):

 if( (i+1) % 1000 == 0 ):

     print("Review %d of %d\n" % (i+1, num_reviews))

 clean_review = review_to_words( test["review"][i] )

 clean_test_reviews.append( clean_review )

test_data_features = vectorizer.transform(clean_test_reviews)

test_data_features = test_data_features.toarray()

result = LR.predict(test_data_features)



output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

# Сохраняем

output.to_csv( "..kaggle/workong/Bag_of_Words_model.csv", index=False, quoting=3 )