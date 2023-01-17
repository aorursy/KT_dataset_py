# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
diamonds_df = pd.read_csv('../input/diamonds.csv')
diamonds_df.drop('Unnamed: 0', axis = 1, inplace = True)
diamonds_df.head()
diamonds_df.info()
diamonds_df.describe()
diamonds_df.drop(diamonds_df[(diamonds_df['x'] == 0) | (diamonds_df['y'] == 0) | (diamonds_df['z'] == 0)].index, inplace = True)
sns.heatmap(diamonds_df.corr(), cmap = 'coolwarm')
#sns.pairplot(diamonds_df, hue = 'cut')
#sns.pairplot(diamonds_df, hue = 'color')
#sns.pairplot(diamonds_df, hue = 'clarity')
sns.boxplot(diamonds_df['color'], diamonds_df['price'])
sns.boxplot(diamonds_df['clarity'], diamonds_df['price'])
sns.boxplot(diamonds_df['cut'], diamonds_df['price'])
sns.distplot(diamonds_df['carat'], bins = 100)
diamonds_df['carat'] = np.log(diamonds_df['carat'])
sns.distplot(diamonds_df['carat'], bins = 100)
sns.distplot(diamonds_df['price'], bins = 100)
diamonds_df['price'] = np.log(diamonds_df['price'])
sns.distplot(diamonds_df['price'], bins = 100)
diamonds_df['volume'] = diamonds_df['x'] * diamonds_df['y'] * diamonds_df['z']
diamonds_df.drop(['x', 'y', 'z'], axis = 1, inplace = True)
df_dummies = pd.get_dummies(diamonds_df[['color', 'clarity', 'cut']])
diamonds_df.drop(['color', 'clarity', 'cut'], axis = 1, inplace = True)
df_new = pd.concat([diamonds_df, df_dummies], axis=1)
plt.figure(figsize = (20,20))
sns.heatmap(df_new.corr(), cmap = 'coolwarm', vmin = -1, vmax = 1, annot = True)
k = 5 #number of variables for heatmap
cols = df_new.corr().nlargest(k, 'price')['price'].index
cm = np.corrcoef(df_new[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values, vmin = -1, vmax= 1)
plt.show()
cols
model_df = df_new[cols]
sns.heatmap(model_df.corr(), cmap = 'coolwarm', vmin=-1, vmax=1, annot = True)
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_real = model_df.drop('price', axis=1)
y_real = model_df['price']
X_train, X_test, y_train, y_test = train_test_split(X_real, y_real, test_size = .3, random_state = 101)
X_train.head()
lm = LinearRegression()
lm.fit(X_train, y_train)
lm.coef_
y_predicted = lm.predict(X_test)
sns.regplot(y_test, y_predicted)
print('MAE: ' + str(metrics.mean_absolute_error(y_test, y_predicted)))
print('MSE: ' + str(metrics.mean_squared_error(y_test, y_predicted)))
print('RMSE: ' + str(np.sqrt(metrics.mean_squared_error(y_test, y_predicted))))
plt.figure(figsize = (10,10))
a = sns.distplot((y_test - y_predicted), bins = 50)