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
import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn import metrics

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv('/kaggle/input/kyphosis/kyphosis.csv')
df.head()
df.sort_values(by='Age',ascending=False)
df.describe()
df.corr()
sns.heatmap((df.corr()))
plt.figure(figsize=(12,9))

sns.distplot(df['Age'], bins=35, color='green')
sns.pairplot(df,hue='Kyphosis', palette='icefire')

plt.legend()
sns.countplot(df['Kyphosis'])
df[df['Age'] > 12].count()
df.groupby(['Kyphosis']).mean()

df.replace('absent', 0, inplace=True)
df.replace('present', 1, inplace=True)
X = df.drop(['Kyphosis'], axis=1)

y = df['Kyphosis']
from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=101, test_size=0.3)
rtree = RandomForestClassifier(n_estimators=300)

rtree.fit(X_train, y_train)
predrtree = rtree.predict(X_test)

predrtree
print(classification_report(y_test, predrtree))

print('\n')

print(confusion_matrix(y_test, predrtree))
log = LogisticRegression()

log.fit(X_train, y_train)
predlog = log.predict(X_test)

predlog
print(classification_report(y_test, predlog))

print('\n')

print(confusion_matrix(y_test, predlog))
print('MAE:', metrics.mean_absolute_error(y_test,predlog))

print('MSE:', metrics.mean_squared_error(y_test,predlog))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test,predlog)))

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)
predknn = knn.predict(X_test)

predknn
print(classification_report(y_test, predknn))

print('\n')

print(confusion_matrix(y_test, predknn))