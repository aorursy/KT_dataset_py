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
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
dataset = pd.read_csv('../input/employee_reviews.csv')
dataset.head()
dataset.info()
dataset["overall-ratings"].unique()
dataset["Liked"] = [1 if i > 2.5 else 0 for i in dataset['overall-ratings']]
dataset['Liked']
data = dataset[['pros','Liked']]
data
sns.countplot(x = data['Liked'],data = data)
import re

from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer

ss = SnowballStemmer('english')
corpus = []

for i in range(0,67529):

    pro = re.sub('[^a-zA_Z]',' ',data['pros'][i])

    pro = pro.lower()

    pro = pro.split()

    pro = [ss.stem(word) for word in pro if word not in set(stopwords.words('english'))]

    pro = ' '.join(pro)

    corpus.append(pro)
corpus[0]
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2000)
x = cv.fit_transform(corpus).toarray()
y = data['Liked']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 40)
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(x_train,y_train)
y_pred = mnb.predict(x_test)
y_train_pred = mnb.predict(x_train)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(classification_report(y_train,y_train_pred))

print(confusion_matrix(y_train,y_train_pred))
print(classification_report(y_test,y_pred))
confusion_matrix(y_test,y_pred)
print('Training Accuracy ---->',accuracy_score(y_train,y_train_pred))

print('Testing Accuracy  ---->',accuracy_score(y_test,y_pred))