# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
import datetime
sns.set()

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def map_container(x):
    if x == '330ml':
        container = 'can'
    elif x == '500ml':
        container = 'glass'
    else:
        container = 'plastic' 
    return container

def map_capacity(x):
    if x == 'can':
        capacity = '330ml'
    elif x == 'glass':
        capacity = '500ml'
    else:
        capacity = '1.5lt'
    return capacity

def graph_demand(df, brands, a):
    fig, axes = plt.subplots(nrows = len(brands), figsize = (15, 25))
    for i in range(len(brands)):
        sns.scatterplot(x=train.quantity[train.brand == brands[i]], y=train.price[train.brand == brands[i]], 
                    hue = train.shop[train.brand == brands[i]], alpha = a, ax = axes[i])
        axes[i].set(title = brands[i], xlabel = 'Quantity', ylabel = 'Price')

    fig.tight_layout()
    plt.show()
train = pd.read_csv('/kaggle/input/predict-demand/train.csv')
test = pd.read_csv('/kaggle/input/predict-demand/test.csv')
train.dropna(how = 'all', inplace = True)
test.dropna(how = 'all', inplace = True)
train.date = pd.to_datetime(train.date)
test.date = pd.to_datetime(test.date)
train.set_index('date', inplace = True)
test.set_index('date', inplace = True)
print(train.info(), '\n')
print(test.info(), '\n')
print(train.tail())
print(test.head())
print(train.describe())
print(test.describe())
pd.DataFrame(train.groupby(['shop', 'long']).size().rename('frequency'))
train.corr()
fig, axes = plt.subplots(ncols = 2, figsize = (18, 7))
sns.distplot(train.quantity, kde = True, ax = axes[0])
sns.distplot(train.price, kde = True, ax = axes[1])
fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (18, 18))
sns.countplot(x = 'city', data = train, ax = axes[0, 0])
sns.countplot(x = 'shop', data = train, ax = axes[0, 1])
sns.countplot(x = 'brand', data = train, ax = axes[1, 0])
sns.countplot(x = 'container', data = train, ax = axes[1, 1])
brands = np.array(train.brand.unique())
graph_demand(train, brands, 0.8)
train_kincola_pivot = train[train.brand == 'kinder-cola'].pivot_table(index = ['price'], values = ['quantity'], aggfunc = np.mean)
train_aducola_pivot = train[train.brand == 'adult-cola'].pivot_table(index = ['price'], values = ['quantity'], aggfunc = np.mean)
train_orpow_pivot = train[train.brand == 'orange-power'].pivot_table(index = ['price'], values = ['quantity'], aggfunc = np.mean)
train_gazoza_pivot = train[train.brand == 'gazoza'].pivot_table(index = ['price'], values = ['quantity'], aggfunc = np.mean)
train_lemboost_pivot = train[train.brand == 'lemon-boost'].pivot_table(index = ['price'], values = ['quantity'], aggfunc = np.mean)

sns.scatterplot(x=train_kincola_pivot.quantity, y=train_kincola_pivot.index)
sns.scatterplot(x=train_aducola_pivot.quantity, y=train_aducola_pivot.index)
sns.scatterplot(x=train_orpow_pivot.quantity, y=train_orpow_pivot.index)
sns.scatterplot(x=train_gazoza_pivot.quantity, y=train_gazoza_pivot.index)
sns.scatterplot(x=train_lemboost_pivot.quantity, y=train_lemboost_pivot.index)
fig, ax = plt.subplots(nrows = 4, figsize = (13, 13))
sns.lineplot(x = train.index, y = train.quantity, ax = ax[0])
sns.lineplot(x = train.index, y = train.price, ax = ax[1])
sns.lineplot(x = train.index, y = train.quantity, hue = train.brand, ax = ax[2])
sns.lineplot(x = train.index, y = train.price, hue = train.brand, ax = ax[3])
ax[1].axhline(np.mean(train.price))
train.loc[train.container.isnull(), 'container'] = train.loc[train.container.isnull(), 'capacity'].apply(map_container)
train.loc[train.capacity.isnull(), 'capacity'] = train.loc[train.capacity.isnull(), 'container'].apply(map_capacity)

test.loc[test.container.isnull(), 'container'] = test.loc[test.container.isnull(), 'capacity'].apply(map_container)
test.loc[test.capacity.isnull(), 'capacity'] = test.loc[test.capacity.isnull(), 'container'].apply(map_capacity)
train.drop(columns = ['id', 'capacity', 'lat', 'long'], inplace = True)
test.drop(columns = ['id', 'capacity', 'lat', 'long'], inplace = True)
print(train.info(), '\n')
print(test.info(), '\n')
train['label'] = 1
test['label'] = 2
temp = pd.concat([train, test])
temp = pd.get_dummies(temp)
train = temp[temp.label == 1]
test = temp[temp.label == 2]
train.drop(columns = ['label'], inplace = True)
test.drop(columns = ['label'], inplace = True)
forest = RandomForestRegressor(n_estimators = 500, random_state = 42)
forest.fit(train.drop(columns = ['quantity']), train['quantity'])
predictions = forest.predict(test.drop(columns = ['quantity']))
mae = metrics.mean_absolute_error(test.quantity, predictions)
mse = metrics.mean_squared_error(test.quantity, predictions)
mape = np.mean(np.abs(test.quantity - predictions) / np.abs(test.quantity))

print('Random Forest Regressor:\n', 16 *'-')
print('Mean Absolute Error: ', mae)
print('Mean Squared Error: ', mse)
print('Mean Absolute Percentage Error: ', 100 * mape, '%')
fig, ax = plt.subplots()
plt.scatter(predictions, test.quantity - predictions, c = 'maroon', marker = '.')
ax.axhline(y = 0, xmin = 0, c = 'r')
ax.set(title = 'Residual Plot', xlabel = 'Predicted Value', ylabel = 'Actual - Predicted')