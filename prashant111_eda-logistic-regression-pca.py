# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# import libraries for plotting

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline





# ignore warnings

import warnings

warnings.filterwarnings('ignore')





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory





# Working with os module - os is a module in Python 3.

# Its main purpose is to interact with the operating system. 

# It provides functionalities to manipulate files and folders.



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
print('# File sizes')

for f in os.listdir('../input'):

    print(f.ljust(30) + str(round(os.path.getsize('../input/' + f) / 1000000, 2)) + 'MB')

%%time



file = ('../input/adult.csv')

df = pd.read_csv(file, encoding='latin-1')
df.shape
df.head()
df.info()
df[df == '?'] = np.nan
df.info()
for col in ['workclass', 'occupation', 'native.country']:

    df[col].fillna(df[col].mode()[0], inplace=True)
df.isnull().sum()
X = df.drop(['income'], axis=1)



y = df['income']
X.head()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
from sklearn import preprocessing



categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']

for feature in categorical:

        le = preprocessing.LabelEncoder()

        X_train[feature] = le.fit_transform(X_train[feature])

        X_test[feature] = le.transform(X_test[feature])
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X.columns)



X_test = pd.DataFrame(scaler.transform(X_test), columns = X.columns)
X_train.head()
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)



print('Logistic Regression accuracy score with all the features: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
from sklearn.decomposition import PCA

pca = PCA()

X_train = pca.fit_transform(X_train)

pca.explained_variance_ratio_

X = df.drop(['income','native.country'], axis=1)

y = df['income']





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)





categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex']

for feature in categorical:

        le = preprocessing.LabelEncoder()

        X_train[feature] = le.fit_transform(X_train[feature])

        X_test[feature] = le.transform(X_test[feature])





X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X.columns)



X_test = pd.DataFrame(scaler.transform(X_test), columns = X.columns)



logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)



print('Logistic Regression accuracy score with the first 13 features: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

X = df.drop(['income','native.country', 'hours.per.week'], axis=1)

y = df['income']





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)





categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex']

for feature in categorical:

        le = preprocessing.LabelEncoder()

        X_train[feature] = le.fit_transform(X_train[feature])

        X_test[feature] = le.transform(X_test[feature])





X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X.columns)



X_test = pd.DataFrame(scaler.transform(X_test), columns = X.columns)



logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)



print('Logistic Regression accuracy score with the first 12 features: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

X = df.drop(['income','native.country', 'hours.per.week', 'capital.loss'], axis=1)

y = df['income']





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)





categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex']

for feature in categorical:

        le = preprocessing.LabelEncoder()

        X_train[feature] = le.fit_transform(X_train[feature])

        X_test[feature] = le.transform(X_test[feature])





X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X.columns)



X_test = pd.DataFrame(scaler.transform(X_test), columns = X.columns)



logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)



print('Logistic Regression accuracy score with the first 11 features: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

X = df.drop(['income'], axis=1)

y = df['income']





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)





categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']

for feature in categorical:

        le = preprocessing.LabelEncoder()

        X_train[feature] = le.fit_transform(X_train[feature])

        X_test[feature] = le.transform(X_test[feature])





X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X.columns)





pca= PCA()

pca.fit(X_train)

cumsum = np.cumsum(pca.explained_variance_ratio_)

dim = np.argmax(cumsum >= 0.90) + 1

print('The number of dimensions required to preserve 90% of variance is',dim)
plt.figure(figsize=(8,6))

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlim(0,14,1)

plt.xlabel('Number of components')

plt.ylabel('Cumulative explained variance')

plt.show()