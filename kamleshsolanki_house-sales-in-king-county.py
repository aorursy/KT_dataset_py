import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.linear_model import LinearRegression
data = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv', parse_dates=['date'])
data = data.drop(['id'], axis = 1)

data.head()
data.info()
data.isna().sum()
df = pd.DataFrame(data.nunique())

df.plot(kind = 'bar')

plt.show()

df.head(len(df))
data.describe()
sns.boxplot(data = data, x = 'price')

plt.show()

sns.distplot(data['price'])

plt.show()
all_data = data[data.price < 2000000]
sns.boxplot(data = all_data, x = 'price')

plt.show()

sns.distplot(all_data['price'])

plt.show()
corr = all_data.corr()

plt.figure(figsize = (15, 7))

sns.heatmap(corr)

plt.show()

corr[['price']].sort_values(by = 'price', ascending = False)
sns.jointplot(data = all_data, x = 'price', y = 'grade')
sns.jointplot(data = all_data, x = 'price', y = 'sqft_living')
sns.jointplot(data = all_data, x = 'price', y = 'sqft_living15')
sns.jointplot(data = all_data, x = 'price', y = 'sqft_above')
sns.jointplot(data = all_data, x = 'price', y = 'bathrooms')
categorical_columns = ['waterfront' , 'view', 'condition', 'grade', 'yr_built', 'yr_renovated', 'zipcode']
all_data.head()
all_data['is_renovated'] = all_data['yr_renovated'].apply(lambda x : x != 0)
sns.catplot(x = 'is_renovated', y = 'price', data = all_data)
all_data['sold_year'] = all_data['date'].apply(lambda x : x.year)

all_data['age'] = all_data['sold_year'] - all_data['yr_built']
sns.catplot(x = 'age', y = 'price', data = all_data)
all_data['renovated_age'] = all_data['sold_year'] - all_data['yr_renovated']

all_data['renovated_age'] = all_data['renovated_age'].apply(lambda x : 10000 if x < 0 else x)
sns.catplot(x = 'renovated_age', y = 'price', data = all_data)
from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
x,y = all_data.drop(['price', 'date'], axis = 1), all_data['price']

x_train, x_val, y_train, y_val = train_test_split(x, y , random_state = 1)
performance_dataframe = pd.DataFrame({'model':[], 'score':[], 'r2_score':[], "mse":[], "mae":[]})
#train model

lr = LinearRegression().fit(x_train, y_train)

y_pre = lr.predict(x_val)

score = lr.score(x_val, y_val)

r_score = r2_score(y_pre, y_val) # r2_score

mae = mean_absolute_error(y_pre, y_val) # r2_score

mse = mean_squared_error(y_pre, y_val) # r2_score



performance_dataframe.loc[performance_dataframe.shape[0]] = ['LinearRegression', score, r_score, mse, mae]
#train model

rf = RandomForestRegressor().fit(x_train, y_train)

y_pre = rf.predict(x_val)

score = rf.score(x_val, y_val)

r_score = r2_score(y_pre, y_val) # r2_score

mae = mean_absolute_error(y_pre, y_val) # r2_score

mse = mean_squared_error(y_pre, y_val) # r2_score



performance_dataframe.loc[performance_dataframe.shape[0]] = ['RandomForestRegressor', score, r_score, mse, mae]
performance_dataframe