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
df = pd.read_csv('/kaggle/input/africa-economic-banking-and-systemic-crisis-data/african_crises.csv')
df.head()
df.isnull().sum()
X = df.drop('banking_crisis',axis=1)

X.head()
y=df.banking_crisis

y.head()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

X.cc3 = le.fit_transform(X.cc3)

X.country = le.fit_transform(X.country)

X.head()
Correlation = X.corr()

Correlation
from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline
plt.figure(figsize=(30,30))

sns.heatmap(Correlation,cmap='coolwarm',annot=True,square=True, fmt ='.3f',annot_kws={'size' : 20})
X.drop(['cc3','case'],axis=1,inplace=True)

X.head()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=1)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix



model = RandomForestClassifier()



# Training the Algorithm

model.fit(X_train,y_train)



# Testing the Algorithm

y_pred = model.predict(X_test)
Accuracy = accuracy_score(y_test,y_pred)*100

print('Accuracy : ',Accuracy,'%')

print('Confusion Matrix : \n')

confusion_matrix(y_test,y_pred)