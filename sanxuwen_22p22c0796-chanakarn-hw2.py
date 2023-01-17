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
df_train = pd.read_csv('../input/titanic/train.csv')

df_test = pd.read_csv('../input/titanic/test.csv')
df_train = df_train.dropna()
X = df_train[['Pclass','Sex','Age','SibSp','Parch','Ticket','Fare','Embarked']]

y = df_train['Survived']
X = pd.get_dummies(X)
from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold

from sklearn.metrics import classification_report

kf = KFold(n_splits=5)

i = 1

for train_index, test_index in kf.split(X):

    X_train, X_val = X.iloc[train_index], X.iloc[test_index]

    y_train, y_val = y.iloc[train_index], y.iloc[test_index]

    

    

    nb = nb.fit(X_train,y_train)

    nn = nn.fit(X_train,y_train)

    dt = dt.fit(X_train,y_train)

    

    pred_nb = nb.predict(X_val)

    pred_nn = nn.predict(X_val)

    pred_dt = dt.predict(X_val)    

    

    print('Fold:'+str(i))

    print('----------------------------------------------------------------------------')

    print('Naive Bayes Report')

    print(classification_report(y_val, pred_nb))

    print('Neural Network Report')

    print(classification_report(y_val, pred_nn))

    print('Decision Tree Report')

    print(classification_report(y_val, pred_dt))

    

    

    

    i = i + 1

    

    

    