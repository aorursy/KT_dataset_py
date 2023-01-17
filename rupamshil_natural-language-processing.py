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
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
dataset=pd.read_csv('/kaggle/input/restaurant-reviews/Restaurant_Reviews.tsv',quoting=3,delimiter='\t')

dataset.head()
import re

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

corpus=[]

for i in range(0,1000):

    reviews=re.sub('[^a-zA-Z]',' ',dataset.iloc[i,0])

    reviews=reviews.lower()

    reviews=reviews.split()

    ps=PorterStemmer()

    all_stopword=stopwords.words('english')

    all_stopword.remove('not')

   

    



    reviews=[ps.stem(word) for word in reviews if not word in set(all_stopword)]

    reviews=' '.join(reviews)

    corpus.append(reviews)

print(corpus)
from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer(max_features=1500 )

X=cv.fit_transform(corpus).toarray()

y=dataset.iloc[:,-1].values

print(len(X[0]))
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.svm import SVC

classifier=SVC(kernel='linear',random_state=0)

classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

print(cm)

accuracy_score(y_test, y_pred)