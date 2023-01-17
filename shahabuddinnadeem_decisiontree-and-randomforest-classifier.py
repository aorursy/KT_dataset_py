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
dataset = pd.read_csv('/kaggle/input/horse-colic/horse.csv')

dataset.head()
import seaborn as sns

import matplotlib.pyplot as plt



sns.countplot(x='outcome',data=dataset,hue='age')
dataset.isnull().sum()
dataset.dtypes.value_counts()
from sklearn.impute import SimpleImputer

imp_mode = SimpleImputer(strategy = 'most_frequent')

for colname in dataset.columns:

    if dataset[colname].dtype == object:

        dataset[colname] = pd.DataFrame(imp_mode.fit_transform(dataset[colname].values.reshape(-1,1)))

dataset.head()
dataset.isnull().sum()
Y = dataset['outcome']

dataset.drop('outcome',axis= 1, inplace = True)

dataset = pd.get_dummies(dataset, prefix_sep='_', drop_first=True,dummy_na=False)

dataset.head()
dataset = pd.DataFrame(imp_mode.fit_transform(dataset), columns= dataset.columns)

dataset.head()
dataset.isnull().sum()
dataset['outcome'] = Y

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

dataset['outcome'] = labelencoder.fit_transform(dataset['outcome'])

dataset.head()
from sklearn.model_selection import train_test_split



X = dataset.iloc[:,:-1].values

y = dataset.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 42)

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)



from sklearn.metrics import accuracy_score,classification_report 

print("Decision Tree Accuracy: ",accuracy_score(y_test, y_pred)*100)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, digits=3))
from sklearn.ensemble import RandomForestClassifier

classifierRF = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 42)

classifierRF.fit(X_train, y_train)

y_predRF = classifierRF.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report

print("Random Forest Accuracy: ",accuracy_score(y_test, y_predRF)*100)

print(classification_report(y_test, y_predRF, digits=3))