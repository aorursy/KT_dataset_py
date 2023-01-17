import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style('whitegrid')
from sklearn.datasets import load_boston

df = load_boston()
print(df.keys())
df.data.shape
df.feature_names
print(df.DESCR)
boston = pd.DataFrame(df.data)
boston.head()
boston.columns = df.feature_names
boston['PRICE'] = df.target
boston.head()
boston.shape
boston.info()
boston.describe().transpose()
boston.isnull().sum()
plt.figure(figsize=(12,8))

sns.heatmap(boston.corr(),cmap='viridis',annot=True,fmt='.2g')
sns.pairplot(data=boston)
sns.distplot(boston['PRICE'],kde=True)
sns.boxplot(boston['PRICE'])
from scipy import stats

import numpy as np

z = np.abs(stats.zscore(boston))

print(z)
threshold = 3

print(np.where(z > 3))
boston_df = boston[(z < 3).all(axis=1)]
boston_df.shape
X = boston_df.drop('PRICE',axis=1)

y = boston_df['PRICE']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
print('Coefficients:',lm.coef_)

print('\n')

print('Intercept:',lm.intercept_)
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')
from sklearn import metrics



print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
sns.distplot((y_test-predictions),bins=50)
coeffecients = pd.DataFrame(lm.coef_,X.columns)

coeffecients.columns = ['Coeffecient']

coeffecients
coeffecients.apply(lambda x: '%.5f' % x, axis=1)