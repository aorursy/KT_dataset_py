import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import numpy as np
housetable = pd.read_csv('../input/usa-housing/USA_Housing.csv')

housetable.head()
housetable.describe()
housetable.info()
housetable.columns


sns.pairplot(housetable, palette="husl", markers='^')
sns.distplot(housetable['Price'], color='m')
sns.heatmap(housetable.corr(), cmap="viridis",annot=True)
X = housetable[housetable.columns[:-2]]

y = housetable['Price']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
X_train.shape
y_train.shape
y_test.shape
X_test.shape
from sklearn.linear_model import LinearRegression

line = LinearRegression()
line.fit(X_train,y_train)
# print the intercept

print(line.intercept_)
coeff_df = pd.DataFrame(line.coef_,X.columns,columns=['Coefficient'])

coeff_df
predictions = line.predict(X_test)
plt.scatter(y_test,predictions, c='g',marker='.')
plt.figure(figsize=(12,4 ))

sns.distplot((y_test-predictions),bins=50,)

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))