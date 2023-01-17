import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import re



import statsmodels.api as sm

import statsmodels.formula.api as smf



from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

from sklearn import metrics



import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv('/kaggle/input/brasilian-houses-to-rent/houses_to_rent.csv')
df.head()
df['hoa'] = df['hoa'].replace('Sem info','0')

df['hoa'] = df['hoa'].replace('Incluso','0')
df['property tax'] = df['property tax'].replace('Sem info','0')

df['property tax'] = df['property tax'].replace('Incluso','0')
df.info()
df.total = df['total'].map(lambda x: re.sub('\D', '', x))

df.hoa = df.hoa.map(lambda x: re.sub('\D', '', x))

df['rent amount'] = df['rent amount'].map(lambda x: re.sub('\D', '', x))

df['property tax'] = df['property tax'].map(lambda x: re.sub('\D', '', x))

df['fire insurance'] = df['fire insurance'].map(lambda x: re.sub('\D', '', x))



# [.map()] Used for substituting each value in a Series with another value

# [\D] Matches any character which is not a decimal digit.



df.head(1)
df.isnull().sum()
df.total = pd.to_numeric(df.total).astype(float)

df.hoa = pd.to_numeric(df.hoa).astype(float)

df['rent amount'] = pd.to_numeric(df['rent amount']).astype(float)

df['property tax'] = pd.to_numeric(df['property tax']).astype(float)

df['fire insurance'] = pd.to_numeric(df['fire insurance']).astype(float)
df.info()
plt.boxplot(df.total)
df[df.total > 50000]
df = df.drop([1269, 3303, 2611, 5627])
df.drop(columns=['total'], inplace=True)
plt.figure(figsize=(10,6))

sns.boxplot(x='rooms', y='rent amount', data=df)
plt.figure(figsize=(10,6))

sns.boxplot(x='bathroom', y='rent amount', data=df)
plt.figure(figsize=(14,10))

minha_ordem = df.groupby('floor')['rent amount'].median().sort_values().iloc[::-1].index

sns.boxplot(x='floor', y='rent amount', data=df, order=minha_ordem)
df.floor = df.floor.map(lambda x: re.sub('-', '10', x))

df.floor = pd.to_numeric(df.floor).astype(float)
df['animal'].value_counts().plot(kind='pie', autopct = '%.2f%%')
df['furniture'].value_counts().plot(kind='pie', autopct = '%.2f%%')
print(pd.DataFrame(df.groupby('furniture')['rent amount'].mean()))

df.groupby('furniture')['rent amount'].mean().plot(kind='barh')
sns.pairplot(df, y_vars=['rent amount'], x_vars=['floor', 'hoa','property tax', 'fire insurance'])
df_s = pd.DataFrame(preprocessing.scale(df[['area', 'rooms', 'bathroom', 'parking spaces', 'floor', 'hoa', 'rent amount',

                             'property tax', 'fire insurance']]))
df[['area', 'rooms', 'bathroom', 'parking spaces', 'floor', 'hoa', 'rent amount', 'property tax', 'fire insurance']] = df_s
df.isna().sum()
df[df.isna().any(axis=1)]
df.drop(index=[6076, 6077, 6078, 6079],inplace=True)
df.drop(columns=['Unnamed: 0'], inplace=True)

# Essa coluna não nos diz nada
df = pd.get_dummies(df, columns=['animal', 'furniture'], drop_first=True)
df.head(1)
X = df.drop(columns=['rent amount'])

y = df['rent amount']
results = sm.OLS(y, X).fit()

print(results.summary())
X = X.drop(columns=['animal_not acept'])

results = sm.OLS(y, X).fit()

print(results.summary())
X = X.drop(columns=['area'])

results = sm.OLS(y, X).fit()

print(results.summary())
X = X.drop(columns=['bathroom'])

results = sm.OLS(y, X).fit()

print(results.summary())
X = X.drop(columns=['city'])

results = sm.OLS(y, X).fit()

print(results.summary())
X = X.drop(columns=['furniture_not furnished'])

results = sm.OLS(y, X).fit()

print(results.summary())
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=34)
model = LinearRegression()

model.fit(X_train, y_train)
predictions = model.predict(X_test)
plt.scatter(y_test,predictions)
score = r2_score(y_test, predictions)

print(score)
print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))