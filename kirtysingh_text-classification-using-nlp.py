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
#importing the libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
#importing the dataset

data=pd.read_csv("../input/spam-text-message-classification/SPAM text message 20170820 - Data.csv")

#Cleaning Of Text

import re

import nltk

nltk.download("stopwords")

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer



corpus=[]

for i in range(0,5572):

    message=re.sub('[^a-zA-Z]',' ',data['Message'][i])

    message=message.lower()

    message=message.split()

    ps=PorterStemmer()

    message=[ps.stem(word) for word in message if not word in set(stopwords.words('english'))]

    message=' '.join(message)

    corpus.append(message)

    
#Creating Bag of words

from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer(max_features=6200)

X=cv.fit_transform(corpus).toarray()

data['Category']=np.where(data['Category']=='spam',1,0)

y=data.iloc[:,0].values
#Splitting the data

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#Building the model

from sklearn.naive_bayes import GaussianNB

classifier=GaussianNB()

classifier.fit(X_train,y_train)



y_pred=classifier.predict(X_test)



from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test, y_pred)
cm
classifier.score(X_test,y_test)