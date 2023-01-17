# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings  

warnings.filterwarnings("ignore")   # ignore warnings



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/employee_reviews.csv')

data.head()
data.info()
data.iloc[:,9:10]
data["overall_ratings"]=data["overall-ratings"]
data['overall_ratings'].unique()
# Comprehension list



data["Liked"] = [1 if i >=3 else 0 for i in data.overall_ratings]

data.tail()
part1 = data.iloc[:,-1:]

part1
part2 = data.iloc[:,6:7]

part2
comment = pd.concat([part2,part1],axis =1,ignore_index =True) 

comment
comment.columns = ['Review', 'Liked']

comment
import re

com = re.sub('[^a-zA-Z]', ' ', comment['Review'][1])

com
# Conversion all letters to lower case 

com = com.lower()

com
# Splitting word by word 

com = com.split()

com
# Loading stopwords



import nltk

from nltk.corpus import stopwords

stopwords_en = stopwords.words('english')

print(stopwords_en)
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
# Cleaning the stop words

# In here, we use set function because set is unordered and same element within set passes only one time.

com = [ps.stem(word) for word in com if not word in set(stopwords.words('english'))]

com
com = ' '.join(com)

com
result = []

for i in range(67529):

    com = re.sub('[^a-zA-Z]', ' ', comment['Review'][i])

    com = com.lower()

    com = com.split()

    com = [ps.stem(word) for word in com if not word in set(stopwords.words('english'))]

    com = ' '.join(com)

    result.append(com)
result
from sklearn.feature_extraction.text import CountVectorizer



### We take most used 2000 words 

cv = CountVectorizer(max_features=2000)



X = cv.fit_transform(result).toarray()

X
y = comment.iloc[:,1].values

y
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train, y_train)



y_pred = gnb.predict(X_test)

y_pred
from sklearn.metrics import confusion_matrix



cm = confusion_matrix(y_test, y_pred)

cm