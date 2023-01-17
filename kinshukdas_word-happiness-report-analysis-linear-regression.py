import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data = pd.read_csv('../input/world-happiness-report-2019.csv')
data.head()
data.describe()
data.isnull().sum()
data.columns
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='median', axis=0)
data.iloc[:, 3:11] = imputer.fit_transform(data.iloc[:, 3:11])
data.isnull().sum()
fig, ax = plt.subplots(figsize=(10,10))

sns.heatmap(data.corr(),ax=ax,annot=True,linewidth=0.05,fmt='.2f')
data.plot(x='Positive affect', y='Negative affect', kind='scatter')

plt.xlabel('Positive affect')

plt.ylabel('Negative affect')
data.plot(x='Positive affect', y='Freedom', kind='scatter')

plt.xlabel('Positive affect')

plt.ylabel('Freedom')
data.plot(x='Positive affect', y='Generosity', kind='scatter')

plt.xlabel('Positive affect')

plt.ylabel('Generosity')
data.plot(x='Positive affect', y='Corruption', kind='scatter')

plt.xlabel('Positive affect')

plt.ylabel('Corruption')
data.plot(x='Ladder', y='SD of Ladder', kind='scatter')

plt.xlabel('Ladder')

plt.ylabel('SD of Ladder')
X = data.drop(['Country (region)', 'Healthy life\nexpectancy'], axis=1)

y = data['Healthy life\nexpectancy']
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
linreg = LinearRegression()

linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)

print('r2 score: ', r2_score(y_test, y_pred))