import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas_profiling

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression,LogisticRegression

from sklearn.neighbors import KNeighborsRegressor

from xgboost import XGBRFRegressor

from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('/kaggle/input/brasilian-houses-to-rent/houses_to_rent_v2.csv')

df.head()
df.dtypes
sns.countplot(df['city'])
sns.countplot(df['bathroom'])
sns.countplot(df['animal'])
df['floor'].value_counts()
df['floor'] = df['floor'].replace('-',0)

df['floor'].value_counts()



df['floor'] = pd.to_numeric(df['floor'])
df.dtypes
sns.countplot(df['furniture'])
from pandas_profiling import ProfileReport

ProfileReport(df)
df.describe()
df.corr()
sns.heatmap(df.corr(),cmap='viridis')
sns.pairplot(df)
plt.plot(df['hoa (R$)'],df['total (R$)'],'o')
df.sort_values('hoa (R$)',ascending=False).head(10)

plt.plot(df['hoa (R$)'],df['total (R$)'],'o')

df = df.drop([255, 6979 , 6230, 2859,2928])

df.sort_values('hoa (R$)',ascending=False).head(10)
plt.plot(df['hoa (R$)'],df['total (R$)'],'o')
df.sort_values(('total (R$)'),ascending=False).head(5)
df = df.drop([6645,2182])  #removing the outliers

df.sort_values(('total (R$)'),ascending=False).head(5)
plt.plot(df['total (R$)'],df['hoa (R$)'],'o')
plt.plot(df['rent amount (R$)'], df['fire insurance (R$)'], 'o')


df.sort_values(('rent amount (R$)'),ascending=False).head(5)
df = df.drop([7748])

plt.plot(df['rent amount (R$)'], df['fire insurance (R$)'], 'o')


categorical = df.dtypes=='object'

categorical_cols = df.columns[categorical].tolist()

print(categorical_cols)



from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

df[categorical_cols]= df[categorical_cols].apply(lambda col:encoder.fit_transform(col.astype(str)))



df.head()
X = df.drop(columns=['rent amount (R$)'])

y = df['rent amount (R$)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
knn =  KNeighborsRegressor()

model = knn.fit(X_train,y_train)

y_pred = model.predict(X_test)

print(np.sqrt(mean_squared_error(y_test,y_pred)))
lr = LinearRegression()

model = lr.fit(X_train,y_train)

y_pred = model.predict(X_test)

print(np.sqrt(mean_squared_error(y_test,y_pred)))
from sklearn.linear_model import Ridge,Lasso

ridge = Ridge(alpha=2.0)

model = ridge.fit(X,y)

y_pred = model.predict(X_test)

print(np.sqrt(mean_squared_error(y_test,y_pred)))
from sklearn.linear_model import Ridge,Lasso

lasso = Lasso(alpha=1.0)

model = lasso.fit(X,y)

y_pred = model.predict(X_test)

print(np.sqrt(mean_squared_error(y_test,y_pred)))
xgb = XGBRFRegressor()

model = xgb.fit(X_train,y_train)

y_pred = model.predict(X_test)

print(np.sqrt(mean_squared_error(y_test,y_pred)))
log = LogisticRegression()

model = log.fit(X_train,y_train)

y_pred = model.predict(X_test)

print(np.sqrt(mean_squared_error(y_test,y_pred)))