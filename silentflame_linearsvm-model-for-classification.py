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
# All library imports

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC

from sklearn import metrics 

from sklearn.metrics import confusion_matrix,classification_report
# reading the data

data= pd.read_csv('../input/sms-spam-collection-dataset/spam.csv',encoding='ISO-8859-1')
data.head()
# Non useful last 3 columns, hence need to remove

data.drop(columns=['Unnamed: 2', 'Unnamed: 3','Unnamed: 4'],inplace=True)

data.rename({'v1': 'labels', 'v2': 'messages'}, axis=1, inplace=True)

data.head()
data.describe()
# can remove the duplicate messages column, let's move ignoring that.

data['length']=data['messages'].apply(len)

data.head()
data['length'].plot(bins=50,kind='hist')

plt.ioff()
data.hist(column='length',by='labels',bins=50,figsize=(10,4))

plt.ioff()
X_train,X_test,y_train,y_test=train_test_split(data['messages'],data['labels'],test_size=0.2,random_state=42)
vectorizer=TfidfVectorizer()

tfidf_vect = vectorizer.fit(data['messages'])

X_train_tfidf = tfidf_vect.transform(X_train)

X_test_tfidf = tfidf_vect.transform(X_test)
clf=LinearSVC()

clf.fit(X_train_tfidf,y_train)
pred = clf.predict(X_test_tfidf)
print(confusion_matrix(y_test,pred))

print(classification_report(y_test,pred))
metrics.accuracy_score(y_test,pred)