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
import warnings

warnings.filterwarnings('ignore')
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import IsolationForest
df = pd.read_excel("/kaggle/input/wholesale-superstore-dataset/Sample - Superstore.xls")

df.head()
df['Sales'].describe()
plt.scatter(range(df.shape[0]), np.sort(df['Sales'].values))

plt.xlabel('index')

plt.ylabel('Sales')

plt.title("Sales distribution")

sns.despine()
sns.distplot(df['Sales'])

plt.title("Distribution of Sales")

sns.despine()
print("Skewness: %f" % df['Sales'].skew())

print("Kurtosis: %f" % df['Sales'].kurt())
df['Profit'].describe()
plt.scatter(range(df.shape[0]), np.sort(df['Profit'].values))

plt.xlabel('index')

plt.ylabel('Profit')

plt.title("Profit distribution")

sns.despine()
sns.distplot(df['Profit'])

plt.title("Distribution of Profit")

sns.despine()
print("Skewness: %f" % df['Profit'].skew())

print("Kurtosis: %f" % df['Profit'].kurt())
isolation_forest = IsolationForest(n_estimators=100)

isolation_forest.fit(df['Sales'].values.reshape(-1, 1))



xx = np.linspace(df['Sales'].min(), df['Sales'].max(), len(df)).reshape(-1,1)

anomaly_score = isolation_forest.decision_function(xx)

outlier = isolation_forest.predict(xx)



plt.figure(figsize=(10,4))

plt.plot(xx, anomaly_score, label='anomaly score')

plt.fill_between(xx.T[0], np.min(anomaly_score), np.max(anomaly_score), 

                 where=outlier==-1, color='r', 

                 alpha=.4, label='outlier region')

plt.legend()

plt.ylabel('anomaly score')

plt.xlabel('Sales')

plt.show();
df.iloc[10]
isolation_forest = IsolationForest(n_estimators=100)

isolation_forest.fit(df['Profit'].values.reshape(-1, 1))



xx = np.linspace(df['Profit'].min(), df['Profit'].max(), len(df)).reshape(-1,1)

anomaly_score = isolation_forest.decision_function(xx)

outlier = isolation_forest.predict(xx)



plt.figure(figsize=(10,4))

plt.plot(xx, anomaly_score, label='anomaly score')

plt.fill_between(xx.T[0], np.min(anomaly_score), np.max(anomaly_score), 

                 where=outlier==-1, color='r', 

                 alpha=.4, label='outlier region')

plt.legend()

plt.ylabel('anomaly score')

plt.xlabel('Profit')

plt.show();
df.iloc[3]
df.iloc[1]