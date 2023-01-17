import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv("../input/housing.csv")

df.head()
df.info()
df.describe()
sns.heatmap(df.corr(), annot=True)
sns.pairplot(df)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics
y = df['MEDV']

X = df[['RM', 'LSTAT', 'PTRATIO']]
lr = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
plt.scatter(y_test, y_pred)
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))

print('MSE:', metrics.mean_squared_error(y_test, y_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
sns.distplot((y_test-y_pred),bins=50);
coeffecients = pd.DataFrame(lr.coef_,X.columns)

coeffecients.columns = ['Coeffecient']

coeffecients