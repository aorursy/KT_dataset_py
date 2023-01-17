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
raw_data=pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv',delimiter=',',encoding='latin-1')

del raw_data['Unnamed: 2']

del raw_data['Unnamed: 3']

del raw_data['Unnamed: 4']

raw_data.head()
text=raw_data['v2']

print(text[0],text[1])
from sklearn.feature_extraction.text import TfidfVectorizer



vectorizer=TfidfVectorizer()

text=vectorizer.fit_transform(text).toarray()
text
data=pd.DataFrame(text)
data.head()
data['Output']=raw_data['v1'].replace({'ham':0,'spam':1})
data.head()
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,confusion_matrix
y=data.pop('Output')

x=data



x_train,x_test,y_train,y_test=train_test_split(x,y,shuffle=True)
model1=GaussianNB()

model1.fit(x_train,y_train)

y_pre=model1.predict(x_test)

print('Report:\n',classification_report(y_pre,y_test))

print('Confusion:\n',confusion_matrix(y_pre,y_test))
model1=MultinomialNB()

model1.fit(x_train,y_train)

y_pre=model1.predict(x_test)

print('Report:\n',classification_report(y_pre,y_test))

print('Confusion:\n',confusion_matrix(y_pre,y_test))
model1=BernoulliNB()

model1.fit(x_train,y_train)

y_pre=model1.predict(x_test)

print('Report:\n',classification_report(y_pre,y_test))

print('Confusion:\n',confusion_matrix(y_pre,y_test))
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt



model=MLPClassifier(verbose=True,hidden_layer_sizes=(80,80,80,),max_iter=2000)

model.fit(x_train,y_train)

y_pre=model.predict(x_test)

print('Report:\n',classification_report(y_pre,y_test))

print('Confusion:\n',confusion_matrix(y_pre,y_test))

plt.plot(model.loss_curve_)

plt.show()