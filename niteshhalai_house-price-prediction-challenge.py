import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/house-price-prediction-challenge/train.csv')

test = pd.read_csv('/kaggle/input/house-price-prediction-challenge/test.csv')

sample_submission = pd.read_csv('/kaggle/input/house-price-prediction-challenge/sample_submission.csv')
train.head(5)
sns.violinplot(data=train, y='TARGET(PRICE_IN_LACS)')

plt.show()
train['TARGET(PRICE_IN_LACS)'].describe()
train[train['TARGET(PRICE_IN_LACS)']>3999]
train[train['SQUARE_FT']>10000000]
f, axes = plt.subplots(1,1,figsize=(15,5))

sns.scatterplot(data=train, x='SQUARE_FT', y='TARGET(PRICE_IN_LACS)')

plt.show()
f, axes = plt.subplots(1,2,figsize=(15,5))

sns.scatterplot(data=train[train['SQUARE_FT']<399999], x='SQUARE_FT', y='TARGET(PRICE_IN_LACS)', ax=axes[0])

sns.scatterplot(data=train[train['SQUARE_FT']>399999], x='SQUARE_FT', y='TARGET(PRICE_IN_LACS)', ax=axes[1])

plt.show()
f, axes = plt.subplots(1,2,figsize=(15,5))



sns.countplot(data=train, x='POSTED_BY', ax=axes[0])

sns.violinplot(data=train, x='POSTED_BY', y='TARGET(PRICE_IN_LACS)', ax=axes[1])

plt.show()
df = pd.read_csv('/kaggle/input/house-price-prediction-challenge/train.csv')



from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso



categorical_columns = []



for column in df.columns:

    if df[column].dtype == 'object':

        categorical_columns.append(column)

        

df = pd.get_dummies(df,columns=categorical_columns, dtype=int, drop_first=True)

df.fillna(0, inplace=True)



y = df['TARGET(PRICE_IN_LACS)']

X = df.drop(labels = ['TARGET(PRICE_IN_LACS)'], axis = 1)



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)



models = [DecisionTreeRegressor(), LinearRegression(), Ridge(),  Lasso()]



for model in models:

    

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)



    from sklearn import metrics

    print('Model:', model)

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))

    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    print('r2_score:', metrics.r2_score (y_test, y_pred))

    print('-------------------------------------')
df = pd.read_csv('/kaggle/input/house-price-prediction-challenge/train.csv')

df.drop(labels=['LONGITUDE','LATITUDE'],axis=1, inplace=True)



from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso



categorical_columns = []



for column in df.columns:

    if df[column].dtype == 'object':

        categorical_columns.append(column)

        

df = pd.get_dummies(df,columns=categorical_columns, dtype=int, drop_first=True)

df.fillna(0, inplace=True)



y = df['TARGET(PRICE_IN_LACS)']

X = df.drop(labels = ['TARGET(PRICE_IN_LACS)'], axis = 1)



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)



models = [DecisionTreeRegressor(), LinearRegression(), Ridge(),  Lasso()]



for model in models:

    

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)



    from sklearn import metrics

    print('Model:', model)

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))

    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    print('r2_score:', metrics.r2_score (y_test, y_pred))

    print('-------------------------------------')
df = pd.read_csv('/kaggle/input/house-price-prediction-challenge/train.csv')

df.drop(labels=['LONGITUDE','LATITUDE', 'ADDRESS'],axis=1, inplace=True)



from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso



categorical_columns = []



for column in df.columns:

    if df[column].dtype == 'object':

        categorical_columns.append(column)

        

df = pd.get_dummies(df,columns=categorical_columns, dtype=int, drop_first=True)

df.fillna(0, inplace=True)



y = df['TARGET(PRICE_IN_LACS)']

X = df.drop(labels = ['TARGET(PRICE_IN_LACS)'], axis = 1)



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)



models = [DecisionTreeRegressor(), LinearRegression(), Ridge(),  Lasso()]



for model in models:

    

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)



    from sklearn import metrics

    print('Model:', model)

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))

    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    print('r2_score:', metrics.r2_score (y_test, y_pred))

    print('-------------------------------------')
df = pd.read_csv('/kaggle/input/house-price-prediction-challenge/train.csv')

df.drop(labels=['LONGITUDE','LATITUDE', 'ADDRESS'],axis=1, inplace=True)



from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso



categorical_columns = []



for column in df.columns:

    if df[column].dtype == 'object':

        categorical_columns.append(column)

        

df = pd.get_dummies(df,columns=categorical_columns, dtype=int, drop_first=True)

df.fillna(0, inplace=True)



y = df['TARGET(PRICE_IN_LACS)']

X = df['SQUARE_FT'].to_numpy().reshape(-1, 1)



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)



models = [DecisionTreeRegressor(), LinearRegression(), Ridge(),  Lasso()]



for model in models:

    

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)



    from sklearn import metrics

    print('Model:', model)

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))

    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    print('r2_score:', metrics.r2_score (y_test, y_pred))

    print('-------------------------------------')
test.head()
test[test['SQUARE_FT']>10000000]
sample_submission.head()