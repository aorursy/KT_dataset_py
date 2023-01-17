# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from sklearn import preprocessing

from itertools import product

import os

pd.set_option('display.float_format', lambda x: '%.2f' % x)


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

print('Done!')
items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
categories = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')
shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')
sales_train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')

print('Updated!')
sales_train['date'] = pd.to_datetime(sales_train.date, format="%d.%m.%Y")
print('shape:', sales_train.shape[0])
sales_train[:5]
sales_train.info()
sales_train.describe()
sns.heatmap(sales_train.corr())
plt.subplots(figsize = (20,6))
plt.subplot(2,3,1)
plt.hist(sales_train['item_price'])
plt.title('Distribution of item price')

plt.subplot(2,3,2)
plt.hist(sales_train['date_block_num'])
plt.title('Distribution of date_block_num')
def timeseriesplot(ejeY, titulo, etiquetaY, Color):
    tg = sales_train.groupby("date_block_num")[ejeY].sum()
    plt.figure(figsize = (20,6))
    plt.title(titulo)
    plt.xlabel('Periodo'), plt.ylabel(etiquetaY)
    plt.plot(tg, color = Color)
    
timeseriesplot("item_price", "Precio de items por periodo", 'Precio', "blue")
timeseriesplot("item_cnt_day", "suma de items por periodo", 'Cantidad', "red")
plt.figure(figsize=(20,5))
sns.countplot(sales_train['date_block_num'],palette="pastel")
plt.title('N° Ventas por mes')
plt.xlabel('Blocks of months (Periods)'), plt.ylabel('N° Compras')
plt.show()
sales_train.isna().sum()
def outliers(df):
    plt.figure(figsize=(20,4))
    plt.subplot2grid((2,3),(0,0))
    sns.boxplot(df.item_price)

    plt.subplot2grid((2,3),(0,1))
    sns.boxplot(df.item_cnt_day)

outliers(sales_train)
train = sales_train[sales_train['item_price'] < 100000]
train = train[train['item_cnt_day'] < 1000] # Esto no es todo, aun hay valores negativos con los cuales lidiar

# Podemos verificar si hay valores negativos con la siguiente función:

#      train['item_price'].nsmallest(5)
#      train['item_cnt_day'].nsmallest(5) 

train = train[train['item_price'] > 0]
train = train[train['item_cnt_day'] > 0]
outliers(train)
train['day'] = train['date'].dt.day
train['month'] = train['date'].dt.month
train['year'] = train['date'].dt.year
train[:5]
plt.figure(figsize=(5,3))
sns.countplot(train['year'], palette="pastel")
plt.title('Balance por años')
plt.xlabel('Años'), plt.ylabel('Frecuencia')
plt.show()
print('shape:', shops.shape[0])
print('Shop_names:', shops['shop_name'].nunique())
shops[:5]
shop_train_list = list(train['shop_id'].unique())
shop_test_list =  list(test['shop_id'].unique())

x = 0
if (set(shop_test_list).issubset(set(shop_train_list))):
    x = 1
print(x)    
train.loc[train['shop_id'] == 0, 'shop_id'] == 57
test.loc[test['shop_id'] == 0, 'shop_id'] == 57

train.loc[train['shop_id'] == 1, 'shop_id'] = 58
test.loc[test['shop_id'] == 1, 'shop_id'] = 58

train.loc[train['shop_id'] == 10, 'shop_id'] = 11
test.loc[test['shop_id'] == 10, 'shop_id'] = 11

shop_train = train['shop_id'].nunique()
shop_test = test['shop_id'].nunique()

print(shop_train)
print(shop_test)
cities = shops['shop_name'].str.split(' ').map(lambda row: row[0])
cities.unique()
shops['city'] = shops['shop_name'].str.split(' ').map(lambda row: row[0])
shops.loc[shops['city'] == '!Якутск', 'city'] = 'Якутск'

le = preprocessing.LabelEncoder()
shops['city_label'] = le.fit_transform(shops['city'])

shops.drop(['shop_name', 'city'], axis = 1, inplace = True)
shops[:5]
plt.figure(figsize=(20,3))
sns.countplot(shops['city_label'], palette="pastel")
plt.title('TIENDAS POR CIUDAD')
plt.xlabel('Ciudad'), plt.ylabel('Tiendas')
plt.show()
print('shape:', items.shape[0])
items.drop(['item_name'], axis = 1, inplace = True)
items[:5]
item_train = train['item_id'].nunique()
item_test = test['item_id'].nunique()
print('Unicos en train:', item_train)
print('Unicos en test:', item_test)

items_train_list = list(train['item_id'].unique())
items_test_list = list(test['item_id'].unique())

check = 0
if(set(items_test_list).issubset(set(items_train_list))):   # Si B es un subconjunto de A
    check = 1
    
print('check is:', check)

""" Si estan todos en la lista check = 1
        de lo contrario
        No estan todos en la lista...  Entonces si no estan todos no podemos predecir ni miercoles"""

print("Cantidad de items diferentes entre test y train sets")       
len(set(items_test_list).difference(items_train_list))
categories_in_test = items.loc[items['item_id'].isin(sorted(test['item_id'].unique()))].item_category_id.unique()
categories.loc[~categories['item_category_id'].isin(categories_in_test)]
ppal_categories = categories['item_category_name'].str.split('-')

# Feature creation
categories['main_category_id'] = ppal_categories.map(lambda row: row[0])
categories['sub_category_id'] = ppal_categories.map(lambda row: row[1] if len(row) > 1 else row[0].strip())

#LabelEncoding for features before.
categories['main_category_id'] = le.fit_transform(categories['main_category_id'])
categories['sub_category_id'] = le.fit_transform(categories['sub_category_id'])

categories.drop(['item_category_name'], axis = 1, inplace = True)
categories[:10]

def downcast_dtypes(df):

        #Changes column types in the dataframe:  from 64 to 32bits
    
    # Select columns to downcast
    float_columns = [c for c in df if df[c].dtype == "float64"]
    int_columns =   [c for c in df if df[c].dtype == "int64"]
    
    # Downcast
    df[float_columns] = df[float_columns].astype(np.float16)
    df[int_columns]   = df[int_columns].astype(np.int16)
    
    return df
months = train['date_block_num'].unique()

cartesian = []

for month in months:
    
    shops_in_month = train.loc[train['date_block_num'] == month, 'shop_id'].unique()
    items_in_month = train.loc[train['date_block_num'] == month, 'item_id'].unique()

    cartesian.append(np.array(
        list(product(*[shops_in_month, items_in_month, [month]])),
         dtype='int32'))


#Debemos crear un elemento(Variable) que almacene el arreglo cartesiano como un dataframe
cartesian = np.vstack(cartesian)   #
Cols = ['shop_id', 'item_id', 'date_block_num']

# Lo convertiremos en un dataframe para trabajar con el
cartesian_df = pd.DataFrame(cartesian, columns = Cols )
print('shape:', cartesian.shape[0])

cartesian_df
# debido a que las columnas ya estan ubicadas por mes item_cnt_day reemplazamos el nombre a item_cnt_month por conveniencia

x = train.groupby(['shop_id', 'item_id', 'date_block_num'])['item_cnt_day'].sum().rename('item_cnt_month').reset_index()
print('shape:', x.shape[0])
x[:5]
month_g = train.groupby(['shop_id', 'item_id', 'date_block_num', 'item_price'])['item_cnt_day'].sum().rename('item_cnt_month').reset_index()
month_g['revenue'] = month_g.item_price * month_g.item_cnt_month

z = month_g.loc[:, ['date_block_num', 'revenue']]

plt.figure(figsize=(10,4))
sns.barplot(x='date_block_num', y='revenue',data=z, palette='colorblind')
plt.title('Revenue por periodo')
plt.xlabel('Periodo'), plt.ylabel('Revenue')
plt.show()

plt.subplots(figsize = (20,6))
plt.subplot(2,3,1)
plt.hist(month_g['revenue'])
plt.title('Distribution of revenue')


# Juntamos con la función merge el dataframe cartesian_df y x...
data = pd.merge(cartesian_df, x, on=['shop_id', 'item_id', 'date_block_num'], how='left').fillna(0)
data['item_cnt_month'] = np.clip(data['item_cnt_month'], 0, 20)

del x
del cartesian_df
del cartesian
del items_test_list
del items_train_list

data.sort_values(['date_block_num','shop_id','item_id'], inplace = True)
print('shape:', data.shape[0])
data[:10]
test.insert(loc=3, column='date_block_num', value = 34) # En la columna 3 asignamos la date_block_num el valor de prediccion 34
test['item_cnt_month'] = 0 

print('shape:', test.shape[0])
test[:5]
data = data.append(test.drop('ID', axis = 1))
data = pd.merge(data, shops, on=['shop_id'], how='left')
data = pd.merge(data, items, on=['item_id'], how='left')
data = pd.merge(data, categories, on='item_category_id', how='left')

print('shape:', data.shape[0])
data[:5]
def generate_feature(train, months, lag_column):

    for month in months:

        train_shift = train[['date_block_num', 'shop_id', 'item_id', lag_column]].copy()
        train_shift.columns = ['date_block_num', 'shop_id', 'item_id', lag_column+'_lag_'+ str(month)]

        train_shift['date_block_num'] += month

        train = pd.merge(train, train_shift, on=['date_block_num', 'shop_id', 'item_id'], how='left')
    
    return train
del items
del categories
del shops
del test
data = downcast_dtypes(data)
import gc       # Recolector de basura.
gc.collect()
'''
Remember the function def generate_features(train, months, lag_column):
'''
%time
data = generate_feature(data, [1,2,3,4,5,6,12], 'item_cnt_month')
%time
group = data.groupby(['date_block_num','item_id'])['item_cnt_month'].mean().rename('item_month_mean').reset_index()

data = pd.merge(data, group, on=['date_block_num','item_id'], how='left')
data = generate_feature(data, [1,2,3,6,12], 'item_month_mean')

data.drop(['item_month_mean'], axis = 1, inplace = True)
%time
group = data.groupby(['date_block_num','shop_id'])['item_cnt_month'].mean().rename('shop_month_mean').reset_index()

data = pd.merge(data, group, on=['date_block_num','shop_id'], how ='left')
data = generate_feature(data, [1,2,3,6,12], 'shop_month_mean')

data.drop(['shop_month_mean'], axis=1, inplace=True)
%time
group = data.groupby(['date_block_num', 'shop_id', 'item_category_id'])['item_cnt_month'].mean().rename('shop_category_month_mean').reset_index()
data = pd.merge(data, group, on=['date_block_num', 'shop_id', 'item_category_id'], how='left')
data = generate_feature(data, [1, 2], 'shop_category_month_mean')
data.drop(['shop_category_month_mean'], axis=1, inplace=True)
%time
group = data.groupby(['date_block_num', 'main_category_id'])['item_cnt_month'].mean().rename('main_category_month_mean').reset_index()
data = pd.merge(data, group, on=['date_block_num', 'main_category_id'], how='left')

data = generate_feature(data, [1], 'main_category_month_mean')
data.drop(['main_category_month_mean'], axis=1, inplace=True)
%time
group = data.groupby(['date_block_num', 'sub_category_id'])['item_cnt_month'].mean().rename('sub_category_month_mean').reset_index()
data = pd.merge(data, group, on=['date_block_num', 'sub_category_id'], how='left')

data = generate_feature(data, [1], 'sub_category_month_mean')
data.drop(['sub_category_month_mean'], axis=1, inplace=True)
data.tail()
data['month'] = data['date_block_num'] % 12
holiday = {
    0: 6,
    1: 3, 
    2: 2,
    3: 8,
    4: 3,
    5: 3,
    6: 2,
    7: 8,
    8: 4,
    9: 8, 
    10: 5, 
    11: 4,
    }
data['holidays_in_month'] = data['month'].map(holiday)
data = data[data.date_block_num > 11]
data[:10]
def fillnan(df):

    for col in df.columns:
        if ('_lag_' in col) & (df[col].isnull().any()):
            df[col].fillna(0, inplace = True)
    return df

data = fillnan(data)
X_train = data[data['date_block_num'] < 33].drop(['item_cnt_month'], axis = 1)
y_train = data[data['date_block_num'] < 33]['item_cnt_month']

X_val = data[data['date_block_num'] == 33].drop(['item_cnt_month'], axis = 1)
y_val = data[data['date_block_num'] == 33]['item_cnt_month']

X_test = data[data['date_block_num'] == 34].drop(['item_cnt_month'], axis = 1)
import optuna
import lightgbm as lgb
from lightgbm import plot_importance

import sklearn.metrics
from sklearn.metrics import mean_squared_error

print('Done!')
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval  = lgb.Dataset(X_val, y_val, reference=lgb_train)
def objective(trial):
    
    # choose parameters that you want
    
    parameters = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }
    
    model = lgb.train(parameters, 
                      lgb_train,
                      valid_sets=[lgb_train,lgb_eval],
                      early_stopping_rounds=15, #10,
                      verbose_eval=1)
    
    y_pred = model.predict(X_val)
    accuracy = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(accuracy)

    return accuracy
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=5)
 
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)
best_params = study.best_trial.params
print(f'Best trial parameters\n{best_params}')
x = {"objective": "regression",
     "metric"   : "rmse",
     "verbosity": -1,
     "boosting_type": "gbdt"}

best_params.update(x)
best_params
evals_result = {} 

model = lgb.train(best_params,
                  lgb_train,
                  valid_sets=[lgb_train,lgb_eval],
                  evals_result=evals_result,
                  early_stopping_rounds=30, # 20
                  verbose_eval=1,
                  )
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
rmse
print('Plot metrics recorded during training...')
ax = lgb.plot_metric(evals_result, figsize=(10, 5))
submission = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')

prediction = model.predict(X_test)

submission['item_cnt_month'] = prediction
print('shape:', submission.shape[0])
submission[:5]
submission.to_csv('submission2.csv', index = False)
print('That´s all Folks')