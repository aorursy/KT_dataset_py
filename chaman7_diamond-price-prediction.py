import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')
# reading the dataset

df = pd.read_csv('../input/diamonds/diamonds.csv')
# how the data looks

df.head()
df.shape
df.info()
# summary of each numerical attribute

df.describe()
df.isnull().sum()
del df['Unnamed: 0']
# Coorelation analysis

plt.figure(figsize=(15,10))

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
sns.pairplot(df)
# The diamond cut categories

df['cut'].value_counts()
sns.countplot(x='cut', data = df)
sns.boxplot('cut', 'price', data = df)
# The diamond color categories

df['color'].value_counts()
sns.countplot(x='color', data = df)
sns.boxplot('color', 'price', data = df)
# The diamond clarity categories

df['clarity'].value_counts()
sns.countplot(x='clarity', data = df)
sns.violinplot('clarity', 'price', data = df)
df.hist(bins = 50, figsize=(15,10))

plt.show()
print("x == 0 : {}".format((df.x==0).sum()))

print("y == 0 : {}".format((df.y==0).sum()))

print("z == 0 : {}".format((df.z==0).sum()))
df.loc[(df['x']==0) | (df['y']==0) | (df['z']==0)]
df[['x','y','z']] = df[['x','y','z']].replace(0,np.NaN)
df.isnull().sum()
df.dropna(inplace=True)
df.shape
import missingno as msno

msno.matrix(df)
one_hot_encoder =  pd.get_dummies(df)

df = one_hot_encoder
df.dtypes
df.head()
X = df.drop('price', axis = 1)

y = df['price']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)



y_pred = regressor.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error



lr_mae = mean_absolute_error(y_test, y_pred)

lr_mse = mean_squared_error(y_test, y_pred)

lr_r2 = r2_score(y_test, y_pred)

print('Linear Regression')

print('Mean Absolute Error:', lr_mae)

print('Mean Squared Error:', lr_mse)

print('R Squared :', lr_r2)
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 20, random_state = 0)

regressor.fit(X_train, y_train)



y_pred = regressor.predict(X_test)
rf_mae = mean_absolute_error(y_test, y_pred)

rf_mse = mean_squared_error(y_test, y_pred)

rf_r2 = r2_score(y_test, y_pred)

print('Random Forest Regressor')

print('Mean Absolute Error:', rf_mae)

print('Mean Squared Error:', rf_mse)

print('R Squared :', rf_r2)