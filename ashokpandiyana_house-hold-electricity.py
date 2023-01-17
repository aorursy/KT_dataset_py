# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset = pd.read_csv("/kaggle/input/electric-power-consumption-data-set/household_power_consumption.txt", sep=';', header=0, infer_datetime_format=True, parse_dates={'datetime':[0,1]}, index_col=['datetime'])
dataset.head()
dataset.tail()
print(f"The Dataset has {dataset.shape[0]} rows and {dataset.shape[1]} columns")
dataset.columns
dataset.info()
dataset.isnull().sum()
percent_missing = dataset.isnull().sum() * 100 / len(dataset)

missing_value_df = pd.DataFrame({'percent_missing': percent_missing})
missing_value_df
dataset1 = dataset.dropna(how = 'any')
dataset1.shape
dataset.loc[dataset.Sub_metering_3.isnull()].head()
dataset.replace('?', np.nan, inplace=True)
dataset.loc[dataset.Sub_metering_3.isnull()].head()
dataset = dataset.dropna(how = 'all')
for i in dataset.columns:

    dataset[i] = dataset[i].astype('float64')

#dataset = dataset.astype('float32')
dataset.shape
values = dataset.values

dataset['sub_metering_4'] = (values[:,0] * 1000 / 60) - (values[:,4] + values[:,5] + values[:,6])
dataset.dtypes
dataset.describe()
sns.distplot(dataset['Global_active_power'])
sns.distplot(dataset['Global_active_power'],kde=False,bins=30)
sns.distplot(dataset['Global_reactive_power'],kde=False,bins=30)
sns.distplot(dataset['Voltage'],kde=True,bins=30)
sns.distplot(dataset['Global_intensity'],kde=True,bins=30)
dataset.corr()
pearson = dataset.corr(method='pearson')

mask = np.zeros_like(pearson)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(pearson, vmax=1, vmin=0, square=True, cbar=True, annot=True, cmap="YlGnBu", mask=mask);
sns.jointplot( x = 'Global_reactive_power' , y = 'Global_active_power' , data = dataset , kind = 'scatter')
sns.jointplot( x = 'Voltage' , y = 'Global_active_power' , data = dataset , kind = 'scatter')
sns.jointplot( x = 'Global_intensity' , y = 'Global_active_power' , data = dataset , kind = 'scatter')
sns.jointplot( x = 'Sub_metering_1' , y = 'Global_active_power' , data = dataset , kind = 'scatter')
sns.jointplot( x = 'Sub_metering_2' , y = 'Global_active_power' , data = dataset , kind = 'scatter')
sns.jointplot( x = 'Sub_metering_3' , y = 'Global_active_power' , data = dataset , kind = 'scatter')
sns.jointplot( x = 'sub_metering_4' , y = 'Global_active_power' , data = dataset , kind = 'scatter')
X = dataset.iloc[:,[1,3,4,5,6]]

y = dataset.iloc[:,0]
X.head()
y.head()
type(X)
type(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
print('Coefficients: \n', lm.coef_)
predictions = lm.predict( X_test)
plt.scatter(y_test,predictions)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

print ('R Squares value:',metrics.r2_score(y_test, predictions))
from sklearn.linear_model import Lasso
best_alpha = 0.00099



regr = Lasso(alpha=best_alpha, max_iter=50000)

regr.fit(X_train,y_train)
lasso_pred = lm.predict(X_test)
print('MAE:', metrics.mean_absolute_error(y_test,lasso_pred))

print('MSE:', metrics.mean_squared_error(y_test,lasso_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test,lasso_pred)))

print ('R Squares value:',metrics.r2_score(y_test,lasso_pred))