import pandas as pd

import numpy as np  

import matplotlib.pyplot as plt  

import seaborn as sns 

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import r2_score

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
train_dataset = pd.read_csv('../input/codenation-enem2/train.csv', index_col=0)

train_dataset.head()
train_dataset['NU_NOTA_MT'].head()
train_dataset.shape
train_dataset.info()
train_dataset.describe()
test_dataset = pd.read_csv('../input/codenation-enem2/test.csv')

test_dataset.head()
test_dataset.shape
test_dataset.info()
test_dataset.describe()
# Generating the answer dataframe with 'NU_INSCRICAO' variable

answer = pd.DataFrame()

answer['NU_INSCRICAO'] = test_dataset['NU_INSCRICAO']

answer.head()
answer.shape
var = ['NU_IDADE','IN_TREINEIRO','NU_NOTA_CN','NU_NOTA_CH','NU_NOTA_LC','NU_NOTA_REDACAO']

train_dataset[var].corr()
features = ['NU_NOTA_CN','NU_NOTA_CH','NU_NOTA_LC','NU_NOTA_REDACAO']
train_dataset[features].corr()
plt.figure(figsize=(9,6))

plt.title('Train Features')

sns.heatmap(train_dataset[features].corr(), annot=True, cmap='Reds')

plt.xticks(rotation=70)

plt.show()
test_dataset[features].corr()
plt.figure(figsize=(9,6))

plt.title('Test Features')

sns.heatmap(test_dataset[features].corr(), annot=True, cmap='Reds')

plt.xticks(rotation=70)

plt.show()
train_dataset[features].isnull().sum()
train_dataset['NU_NOTA_MT'].isnull().sum()
test_dataset[features].isnull().sum()
train_dataset['NU_NOTA_CN'].fillna(0, inplace=True)

train_dataset['NU_NOTA_CH'].fillna(0, inplace=True)

train_dataset['NU_NOTA_REDACAO'].fillna(0, inplace=True)

train_dataset['NU_NOTA_LC'].fillna(0, inplace=True)

train_dataset['NU_NOTA_MT'].fillna(0, inplace=True)

test_dataset['NU_NOTA_CN'].fillna(0, inplace=True)

test_dataset['NU_NOTA_CH'].fillna(0, inplace=True)

test_dataset['NU_NOTA_REDACAO'].fillna(0, inplace=True)

test_dataset['NU_NOTA_LC'].fillna(0, inplace=True)
train_dataset[features].isnull().sum()
train_dataset['NU_NOTA_MT'].isnull().sum()
test_dataset[features].isnull().sum()
X = train_dataset[features]

X.head()
y = train_dataset['NU_NOTA_MT']

y.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
sc = StandardScaler()

sc.fit(X_train)

X_train = sc.transform(X_train)

X_test = sc.transform(X_test)
lr = LinearRegression()

lr.fit(X_train, y_train)
# Getting predictions

y_pred = lr.predict(X_test)
# Getting r2 score

r2_score(y_test, y_pred)
# Perform Grid-Search

gsc = GridSearchCV(

    estimator=RandomForestRegressor(),

    param_grid={'max_depth': range(3,7), 

                'n_estimators': (50, 100, 500, 1000),

    },

    cv=10, scoring='r2', verbose=0, n_jobs=-1)



grid_result = gsc.fit(X, y)

best_params = grid_result.best_params_

rfr = RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"], random_state=False, verbose=False)



# Perform K-Fold CV

scores = cross_val_score(rfr, X, y, cv=10, scoring='r2')

scores
scores.mean() * 100
rfr.fit(train_dataset[features], train_dataset['NU_NOTA_MT'])
y_pred = rfr.predict(test_dataset[features])

y_pred
answer['NU_NOTA_MT'] = y_pred

answer.head()
answer.describe()
answer.to_csv('answer.csv', index=False, float_format='%.1f')