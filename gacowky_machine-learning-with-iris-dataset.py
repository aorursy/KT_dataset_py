import numpy as np

import pandas as pd

import seaborn as sns

sns.set_palette('husl')

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn import metrics

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



data = pd.read_csv('../input/Iris.csv')
data.head()
data.info()
data.describe()
data['Species'].value_counts()
tmp = data.drop('Id', axis=1)

g = sns.pairplot(tmp, hue='Species', markers='+')

plt.show()
g = sns.violinplot(y='Species', x='SepalLengthCm', data=data, inner='quartile')

plt.show()

g = sns.violinplot(y='Species', x='SepalWidthCm', data=data, inner='quartile')

plt.show()

g = sns.violinplot(y='Species', x='PetalLengthCm', data=data, inner='quartile')

plt.show()

g = sns.violinplot(y='Species', x='PetalWidthCm', data=data, inner='quartile')

plt.show()
X = data.drop(['Id', 'Species'], axis=1)

y = data['Species']

# print(X.head())

print(X.shape)

# print(y.head())

print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred))
print(y_pred)