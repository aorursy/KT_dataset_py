import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Libraries to plot

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
%%time



df = pd.read_csv('/kaggle/input/adult-census-income/adult.csv')
df.shape
df.head()
print(df.info())

df[df == '?'] = np.nan
df.info()
for column in ['workclass', 'occupation', 'native.country']:

    df[column].fillna(df[column].mode()[0], inplace=True)
df.isnull().sum()
X = df.drop(['income'], axis=1)

y = df['income']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
from sklearn import preprocessing

categoric = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']

for feature in categoric:

    le = preprocessing.LabelEncoder()

    X_train[feature] = le.fit_transform(X_train[feature])

    X_test[feature] = le.transform(X_test[feature])
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train))

X_test = pd.DataFrame(scaler.transform(X_test))
X_train.head()
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)



print("Logistic Regression model accuracy score with all features:", accuracy_score(y_pred, y_test))