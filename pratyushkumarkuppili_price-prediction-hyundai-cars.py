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
hyundai_data = pd.read_csv('/kaggle/input/used-car-dataset-ford-and-mercedes/hyundi.csv')
hyundai_data.head()
hyundai_data.isna().sum()
len(hyundai_data)
hyundai_data.info()
hyundai_data.model.value_counts()
hyundai_data.year.value_counts()
hyundai_data['car-age'] = 2020 - hyundai_data['year']
hyundai_data.drop('year', 1, inplace = True)
import matplotlib.pyplot as plt
import seaborn as sns
sns.lineplot(x = 'engineSize', y = 'price', data = hyundai_data)
plt.figure(figsize=(10,8))
sns.scatterplot(x= 'mileage',y = 'price', hue = 'fuelType', data = hyundai_data)
plt.figure(figsize=(10,8))
sns.scatterplot(x= 'mpg',y = 'price',hue = 'fuelType', data = hyundai_data)
plt.figure(figsize=(10,8))
sns.scatterplot(x= 'tax(£)',y = 'price', hue = 'fuelType', data = hyundai_data)
plt.figure(figsize=(10,8))
sns.scatterplot(x= 'car-age',y = 'price', hue = 'fuelType', data = hyundai_data)
plt.figure(figsize=(10,8))
sns.catplot(y = 'model', x= 'price' , data = hyundai_data)
plt.figure(figsize = (20,12))
plt.subplot(2,2,1)
sns.boxplot(hyundai_data.price)
plt.subplot(2,2,2)
sns.boxplot(hyundai_data['tax(£)'])
plt.subplot(2,2,3)
sns.boxplot(hyundai_data.mpg)
plt.subplot(2,2,4)
sns.boxplot(hyundai_data.engineSize)
hyundai_data.describe()
print(hyundai_data.price.quantile(0.995))
print(hyundai_data['tax(£)'].quantile(0.995))
print(hyundai_data.engineSize.quantile(0.995))
print(hyundai_data['mpg'].quantile(0.995))
hyundai_data = hyundai_data[hyundai_data.price < hyundai_data.price.quantile(0.995)]
hyundai_data = hyundai_data[(hyundai_data['tax(£)'] < hyundai_data['tax(£)'].quantile(0.995))& (hyundai_data['tax(£)'] > 0)]
hyundai_data= hyundai_data[(hyundai_data.engineSize < hyundai_data.engineSize.quantile(0.995)) & (hyundai_data.engineSize > 0)]
hyundai_data = hyundai_data[(hyundai_data['mpg'] < hyundai_data['mpg'].quantile(0.995)) & (hyundai_data['mpg'] > 0)]
hyundai_data.reset_index(drop = True, inplace = True)
plt.figure(figsize = (20,12))
plt.subplot(2,2,1)
sns.boxplot(hyundai_data.price)
plt.subplot(2,2,2)
sns.boxplot(hyundai_data['tax(£)'])
plt.subplot(2,2,3)
sns.boxplot(hyundai_data.mpg)
plt.subplot(2,2,4)
sns.boxplot(hyundai_data.engineSize)
hyundai_data.head()
hyundai_data.info()
sns.pairplot(data = hyundai_data)
sns.heatmap(hyundai_data.corr())
hyundai_data.columns
cat_cols = ['model', 'transmission', 'fuelType']
hyundai_cat = hyundai_data[cat_cols]
hyundai_cat = pd.get_dummies(hyundai_cat, drop_first = True)
hyundai_cat.head()
num_cols = ['price','mileage', 'tax(£)','mpg', 'engineSize', 'car-age']
hyundai_num = hyundai_data[num_cols]
hyundai = pd.concat([hyundai_num, hyundai_cat], axis =1)
target_variable = hyundai['price']
hyundai.drop('price',1, inplace = True)
hyundai.head()
cols = hyundai.columns
from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
hyundai_pro_data = pd.DataFrame(minmax.fit_transform(hyundai), columns = cols)
hyundai_pro_data.head()
X = pd.concat([hyundai_pro_data, target_variable],1)
X
y = X['price']
X = X.drop('price',1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 42)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)
y_pred = regression_model.predict(X_test)
y_pred
y_test
rmse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(rmse)
print(r2)
from sklearn.metrics import r2_score
coefficient_of_determination = r2_score(y_test,y_pred, multioutput='variance_weighted')
coefficient_of_determination 
import statsmodels.api as sm
X_train_lm = sm.add_constant(X_train)
lr_1 = sm.OLS(y_train, X_train_lm).fit()
lr_1.params
lr_1.summary()
lr_1.predict(sm.add_constant(X_test))
y_test
