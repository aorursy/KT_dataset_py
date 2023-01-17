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
df = pd.read_csv('/kaggle/input/predicting-a-pulsar-star/pulsar_stars.csv')

df.head()
df.info()
df.describe()
df.target_class.value_counts()
import seaborn as sns
sns.pairplot(df, hue = 'target_class')
y = df.target_class

df.drop(columns='target_class', inplace = True)

df.head()
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

scaler.fit(df)

dfs = scaler.transform(df)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(dfs, y, test_size=0.3)
from sklearn.svm import SVC



clf = SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred))

print('\n')

print(confusion_matrix(y_test, y_pred))
y_test.value_counts()
from sklearn.linear_model import LogisticRegression



clf_log = LogisticRegression()
clf_log.fit(X_train, y_train)
y_pred = clf_log.predict(X_test)
print(classification_report(y_test, y_pred))

print('\n')

print(confusion_matrix(y_test, y_pred))
pd.DataFrame(y_pred).to_csv('prediction.csv', index = False)