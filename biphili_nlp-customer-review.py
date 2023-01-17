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



# Any results you write to the current directory are saved as output.
# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import re

import nltk

from nltk.stem.porter import PorterStemmer
# Importing the dataset

df = pd.read_csv('../input/restaurantreviews/Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
df.head()
#df['Review'][0]
#review=re.sub('[^a-zA-Z]',' ',df['Review'][0])
#review
#review=review.lower()
#review
nltk.download('stopwords')

from nltk.corpus import stopwords

#review=review.split()
#ps=PorterStemmer()
#review
#review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

#review
#review=' '.join(review)
#review
corpus=[]

for i in range(0,1000):

    review=re.sub('[^a-zA-Z]',' ',df['Review'][i])

    review=review.lower()

    review=review.split()

    ps=PorterStemmer()

    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

    review=' '.join(review)

    corpus.append(review)
corpus[0:5]
from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer()

X=cv.fit_transform(corpus).toarray()
X.shape
from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer(max_features=1500)

X=cv.fit_transform(corpus).toarray()
X.shape
y=df.iloc[:,1].values
y.shape
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15)
from sklearn.preprocessing import StandardScaler

sc_X=StandardScaler()

X_train=sc_X.fit_transform(X_train)

X_test=sc_X.fit_transform(X_test)
from sklearn.naive_bayes import GaussianNB

classifier=GaussianNB()

classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)

cm
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))