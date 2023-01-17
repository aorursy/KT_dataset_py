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
import matplotlib.pyplot as plt

import seaborn as sns
dataset = pd.read_csv('../input/Restaurant_Reviews.tsv',delimiter = '\t')
dataset.head()
dataset.describe()
dataset.info()
dataset.isnull().sum()
sns.countplot(x = dataset['Liked'],data = dataset)
dataset[dataset['Liked'] == 1]["Liked"].count()
dataset[dataset['Liked'] == 0]['Liked'].count()
from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer

import re
stemmer = SnowballStemmer('english')

corpus = []

for i in range(0,1000):

    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])

    review = review.lower()

    review = review.split()

    review = [stemmer.stem(word) for word in review if word not in set(stopwords.words('english'))]

    review = ' '.join(review)

    corpus.append(review)
corpus[1]
len(corpus)
corpus[999]
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(corpus).toarray()
x.shape
y = dataset['Liked'].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 17)
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
y_train_pred = classifier.predict(x_train)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(classification_report(y_test,y_pred))
confusion_matrix(y_test,y_pred)
print('Training Accuray --->',accuracy_score(y_train,y_train_pred))

print('Testing Accuray --->',accuracy_score(y_test,y_pred))
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=800)

rf.fit(x_train,y_train)
print(classification_report(y_test,y_pred))
confusion_matrix(y_test,y_pred)
y_train_pred = rf.predict(x_train)
print('Traning Accuracy --->',accuracy_score(y_train,y_train_pred))

print('Testing Accuracy --->',accuracy_score(y_test,y_pred))
from sklearn.svm import SVC
svc = SVC(gamma = 'scale')
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
from sklearn.model_selection import GridSearchCV
parameters = [{'C':[1,10,100,1000],'kernel':['linear']},{'C':[1,10,100,1000],'kernel':['rbf'],'gamma':[1,0.5,0.1,0.01,0.001]}]
gs = GridSearchCV(estimator=SVC(),param_grid=parameters,scoring='accuracy',cv = 10)
gs = gs.fit(x_train,y_train)
gs
y_pred = gs.predict(x_test)
print(classification_report(y_test,y_pred))
confusion_matrix(y_test,y_pred)
y_train_pred = gs.predict(x_train)
print('Training Accuracy --->',accuracy_score(y_train,y_train_pred))

print('Testing Accuracy --->',accuracy_score(y_test,y_pred))