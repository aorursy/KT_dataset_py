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
df = pd.read_csv('../input/heart-disease-prediction-using-logistic-regression/framingham.csv')
df.head()
print("The dataset is consisted of {} entries and {} features".format(df.shape[0], df.shape[1]))
df.info()
df.isnull().sum()
df = df.drop(['education'],axis=1)
df.cigsPerDay.describe()
df['cigsPerDay'].fillna(df['cigsPerDay'].median(), inplace=True)
df.BPMeds.value_counts()
df['BPMeds'].fillna(df['BPMeds'].value_counts().index[0], inplace=True)
df.totChol.describe()
df['totChol'].fillna(df['totChol'].mean(), inplace=True)
df.BMI.describe()
df['BMI'].fillna(df['BMI'].mean(), inplace=True)
df.heartRate.describe()
df['heartRate'].fillna(df['heartRate'].value_counts().index[0], inplace=True)
df.glucose.describe()
df['glucose'].fillna(df['glucose'].mean(), inplace=True)
df.isnull().sum()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from statsmodels.tools import add_constant
import warnings
warnings.filterwarnings('ignore')

X = df.drop(['TenYearCHD'], axis=1)
X = add_constant(X)
y = df['TenYearCHD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

logReg = LogisticRegression().fit(X_train, y_train)

train_pred = logReg.predict(X_train)
test_pred = logReg.predict(X_test)

print('Train set accuracy score:', accuracy_score(y_train, train_pred))
print('Test set accuracy score:', accuracy_score(y_test, test_pred))
import statsmodels.api as sm

result = sm.Logit(y, X).fit()
result.summary()
X.drop(['currentSmoker'], axis=1, inplace=True)

result = sm.Logit(y, X).fit()
result.summary()
X.drop(['BMI'], axis=1, inplace=True)

result = sm.Logit(y, X).fit()
result.summary()
X.drop(['heartRate'], axis=1, inplace=True)

result = sm.Logit(y, X).fit()
result.summary()
X.drop(['diaBP'], axis=1, inplace=True)

result = sm.Logit(y, X).fit()
result.summary()
X.drop(['diabetes'], axis=1, inplace=True)

result = sm.Logit(y, X).fit()
result.summary()
X.drop(['BPMeds'], axis=1, inplace=True)

result = sm.Logit(y, X).fit()
result.summary()
X.drop(['prevalentHyp'], axis=1, inplace=True)

result = sm.Logit(y, X).fit()
result.summary()
X.drop(['totChol'], axis=1, inplace=True)

result = sm.Logit(y, X).fit()
result.summary()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

logReg = LogisticRegression().fit(X_train, y_train)
train_pred = logReg.predict(X_train)
test_pred = logReg.predict(X_test)

print('New train set accuracy:', accuracy_score(y_train, train_pred))
print('New test set accuracy:', accuracy_score(y_test, test_pred))
confusion_matrix(y_test, test_pred)