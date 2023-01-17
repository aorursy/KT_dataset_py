# numerical analysis

import numpy as np

# storing and processing in dataframes

import pandas as pd

# basic plotting

import matplotlib.pyplot as plt

# advanced plotting

import seaborn as sns



# splitting dataset into train and test

from sklearn.model_selection import train_test_split

# scaling features

from sklearn.preprocessing import MinMaxScaler, StandardScaler

# k nearest neighbors model

from sklearn.neighbors import KNeighborsRegressor

# linear regression model

from sklearn.linear_model import LinearRegression

# decision tree model

from sklearn.tree import DecisionTreeRegressor

# random foreset regressor model

from sklearn.ensemble import RandomForestRegressor

# evaluation metrics

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# plot style

sns.set_style('whitegrid')

# color palettes

pal = ['orangered', 'black']
# read data

df = pd.read_csv('../input/Automobile_data.csv', na_values='?')



# first few rows

df.head()
# columns names

df.columns
# no. of rows and columns

df.shape
# consise summary of dataframe

df.info()
# descriptive statistics

df.describe(include='all')
# no. of na values in each columns

df.isna().sum()
# plot figure

plt.figure(figsize=(12, 5))

# plot missing values heatmap

sns.heatmap(df.isna(), cbar=False, cmap='Set3')

# title

plt.title('Missing values in each columns')

# x-ticks rotation

# plt.xticks(rotation='80')

# show the plot

plt.show()
for col in df.columns:

    if df[col].nunique() <= 7:

        print(col)

        print("="*len(col))

        print(df[col].value_counts())

        print('dtype: ', df[col].dtype)

        print('\n')
cat_cols = ['fuel-type', 'aspiration', 'num-of-doors', 'body-style', 

            'drive-wheels', 'engine-location', 'fuel-system']



for col in cat_cols:

    df[col] = df[col].astype('category')
num_cols = []

cat_cols = []



for i in df.columns[:]:

    if(df[i].dtype=='object'):

        cat_cols.append(i)

    else:

        num_cols.append(i)

        

print(num_cols)

print(cat_cols)
for i in cat_cols:

    if((df[i] == '?').sum()>0):

        print(i, (df[i] == '?').sum())
df.drop('normalized-losses', axis=1, inplace=True)

cat_cols.remove('normalized-losses')
# first replace '?' with np.nan

df.replace('?', np.nan, inplace=True)

for i in cat_cols:

    if((df[i] == '?').sum()>0):

        print(i, (df[i] == '?').sum())

    

# then drop rows with na

df.dropna(axis=0, inplace=True)
for i in ['bore', 'horsepower', 'stroke', 'peak-rpm', 'price']:

    df[i] = df[i].astype('float64')

    num_cols.append(i)

    cat_cols.remove(i)

    

df['symboling'] = df['symboling'].astype('object')

cat_cols.append('symboling')

num_cols.remove('symboling')
for i in num_cols[:-1]:

    plt.figure(figsize=(8, 5))

    sns.regplot(x=i, y='price', data=df)

    plt.plot()
for i in cat_cols:

    plt.figure()

    sns.boxplot(x=i , y='price', data=df)

    plt.plot()
plt.figure(figsize=(12, 10))

sns.heatmap(df.corr(), annot=True, cmap='RdBu', vmin=-1, vmax=1)

plt.plot()
df = df.drop(['height', 'stroke', 'compression-ratio', 'peak-rpm', 'num-of-doors'], axis=1)

df.head()



# removing those name from the list num_cols

for i in ['height', 'stroke', 'compression-ratio', 'peak-rpm']:

    num_cols.remove(i)

    

# removing those name from the list cat_cols

cat_cols.remove('num-of-doors')
sc = StandardScaler()



for i in num_cols[:-1]:

    df[i] = df[i].astype('float64')

    df[i] =  sc.fit_transform(df[i].values.reshape(-1,1))

    

df.head()
df = pd.get_dummies(df, drop_first=True)

df.head()
X = df.drop('price', axis=1)

y = df['price']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model = LinearRegression()

model.fit(X_train, y_train)



pred = model.predict(X_test)



print(mean_absolute_error(y_test, pred))

print(mean_squared_error(y_test, pred))

print(r2_score(y_test, pred))



fig, ax = plt.subplots(1, 3, figsize=(24, 5))



sns.scatterplot(x=pred, y=y_test, ax=ax[0])

plt.xlabel("Predicted values")

plt.ylabel("Actual values")

plt.plot()



sns.distplot(pred-y_test, ax=ax[1])

plt.xlabel("Difference")

plt.ylabel("Count")

plt.plot()



sns.scatterplot(x=pred, y=y_test-pred, ax=ax[2])

plt.xlabel("Predicted values")

plt.ylabel("Difference")

plt.plot()
model = DecisionTreeRegressor()

model.fit(X_train, y_train)



pred = model.predict(X_test)



print(mean_absolute_error(y_test, pred))

print(mean_squared_error(y_test, pred))

print(r2_score(y_test, pred))



fig, ax = plt.subplots(1, 3, figsize=(24, 5))



sns.scatterplot(x=pred, y=y_test, ax=ax[0])

plt.xlabel("Predicted values")

plt.ylabel("Actual values")

plt.plot()



sns.distplot(pred-y_test, ax=ax[1])

plt.xlabel("Difference")

plt.ylabel("Count")

plt.plot()



sns.scatterplot(x=pred, y=y_test-pred, ax=ax[2])

plt.xlabel("Predicted values")

plt.ylabel("Difference")

plt.plot()
model = RandomForestRegressor()

model.fit(X_train, y_train)



pred = model.predict(X_test)



print(mean_absolute_error(y_test, pred))

print(mean_squared_error(y_test, pred))

print(r2_score(y_test, pred))



fig, ax = plt.subplots(1, 3, figsize=(24, 5))



sns.scatterplot(x=pred, y=y_test, ax=ax[0])

plt.xlabel("Predicted values")

plt.ylabel("Actual values")

plt.plot()



sns.distplot(pred-y_test, ax=ax[1])

plt.xlabel("Difference")

plt.ylabel("Count")

plt.plot()



sns.scatterplot(x=pred, y=y_test-pred, ax=ax[2])

plt.xlabel("Predicted values")

plt.ylabel("Difference")

plt.plot()