# Setting up libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn

from sklearn.preprocessing import LabelEncoder

from itertools import product

import gc

from xgboost import XGBRegressor

from xgboost import plot_importance



def plot_features(booster, figsize):    

    fig, ax = plt.subplots(1,1,figsize=figsize)

    return plot_importance(booster=booster, ax=ax)
# Importing the data



import os

print(os.listdir("../input"))



sales_train = pd.read_csv('../input/sales_train.csv')

test_df = pd.read_csv('../input/test.csv').set_index('ID')

item_cat = pd.read_csv('../input/item_categories.csv')

items = pd.read_csv('../input/items.csv')

shops = pd.read_csv('../input/shops.csv')
sales_train.shape
sales_train.head(10)
sales_train.info()
sales_train.describe()
sales_train['item_price'].isnull().values.any()
plt.figure(figsize = (10,4))

plt.xlim(sales_train.item_price.min(), sales_train.item_price.max()*1.1)

sns.boxplot(x = sales_train.item_price)
price_outlier = sales_train.loc[sales_train['item_id'] == 6066]

price_outlier.head()
# Tem que juntar com o DF de items para pegar o item_name e dar certo.



#price_item_outlier = sales_train[sales_train['item_name'].str.contains('Radmin')]

#price_item_outlier.head()
# Depende do output acima



#price_item_outlier.groupby(['item_name'])['item_price'].mean()



# Output dos códigos em cima está descrito na introdução de 'Coluna | item_price'
price_neg = sales_train.loc[sales_train['item_price'] <= 0]

price_neg.head()
price_item_neg = sales_train.loc[(sales_train['item_id'] == 2973) & (sales_train['shop_id'] == 32)]

price_item_neg.head(10)
# Substituí o valor de -1 pelo valor do produto para a loja em questão

sales_train['item_price'] = sales_train['item_price'].replace(-1, 2499)
# REMOVER PRICE OUTLIER

sales_train = sales_train[sales_train.item_price < 100000]
sales_train['item_cnt_day'].isnull().values.any()
plt.figure(figsize = (10,4))

plt.xlim(-100, 3000)

sns.boxplot(x = sales_train.item_cnt_day)
cnt_big = sales_train.loc[sales_train['item_cnt_day'] > 525]

cnt_big['item_cnt_day'].value_counts()
cnt_big.head(10)
cnt_item_outlier = sales_train.loc[sales_train['item_id'] == 11373]

cnt_item_outlier.head()
# REMOVER CNT OUTLIER

sales_train = sales_train[sales_train.item_cnt_day < 1001]
# 'date' column

#transactions['date'] = pd.to_datetime(transactions['date'], format = "%d.%m.%Y")

#transactions['dt_year'] = transactions['date'].dt.year

#transactions['dt_month'] = transactions['date'].dt.month

#transactions['dt_day'] = transactions['date'].dt.day

#transactions.head(5)

#transactions.groupby(['dt_year', 'dt_month'])['dt_day'].nunique()



# 'date_block_column'

#transactions.groupby(['dt_year', 'dt_month'])['date_block_num'].value_counts()
# 5.100 items such that 363 are new items (5.100 X 42 shops = 214.200)



len(list(set(test_df.item_id) - set(test_df.item_id).intersection(set(sales_train.item_id)))), len(list(set(test_df.item_id))), len(test_df)
test_df.head(10)
test_df.info()
test_df.describe()
item_cat.shape
item_cat.head(30)
item_cat.info()
item_cat.describe()
# Split in 'Category' and 'Subcategory':



item_cat['split'] = item_cat['item_category_name'].str.split('-')

item_cat['item_macro_categ'] = item_cat['split'].map(lambda x: x[0].strip())

item_cat['item_macro_categ_code'] = LabelEncoder().fit_transform(item_cat['item_macro_categ'])

# if subtype is nan then type

item_cat['item_subcateg'] = item_cat['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())

item_cat['item_subcateg_code'] = LabelEncoder().fit_transform(item_cat['item_subcateg'])



set_macrocateg = set(item_cat['item_macro_categ'])

set_subcateg = set(item_cat['item_subcateg'])

item_cat = item_cat[['item_category_id','item_macro_categ_code', 'item_subcateg_code']]



print('Macrocategory'), print(set_macrocateg), print('----------'), print('Subcategory'), print(set_subcateg)
item_cat.head(10)
items.shape
items.head(10)
items.info()
items.describe()
items.drop(['item_name'], axis=1, inplace=True)

items = items.join(item_cat.set_index('item_category_id'), on = 'item_category_id')

items.head()
shops.shape
shops.head(60)
# Ajustando shoppings que estão repetidos (o ajuste é pelo código direto no train e test set)



# Якутск Орджоникидзе, 56

sales_train.loc[sales_train.shop_id == 0, 'shop_id'] = 57

test_df.loc[test_df.shop_id == 0, 'shop_id'] = 57

# Якутск ТЦ "Центральный"

sales_train.loc[sales_train.shop_id == 1, 'shop_id'] = 58

test_df.loc[test_df.shop_id == 1, 'shop_id'] = 58

# Жуковский ул. Чкалова 39м²

sales_train.loc[sales_train.shop_id == 10, 'shop_id'] = 11

test_df.loc[test_df.shop_id == 10, 'shop_id'] = 11
shops.info()
shops.describe()
# City

shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'

shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])

shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'

shops['city_code'] = LabelEncoder().fit_transform(shops['city'])



# Store type

def store_type(x):

    if 'ТЦ' in x:

        z = 0

    elif 'ТРЦ' in x:

        z = 1    

    elif 'ТРК' in x:

        z = 2

    elif 'ТК' in x:

        z = 3

    elif 'ул' in x:

        z = 4

    elif 'Интернет-магазин' in x:

        z = 5

    elif 'Цифровой склад' in x:

        z = 6

    else:

        z = 7

    return z



shops['store_type'] = shops['shop_name'].apply(store_type).astype('int64')



shops = shops[['shop_id','city_code', 'store_type']]

shops.head(10)
# 'date' column

#transactions['date'] = pd.to_datetime(transactions['date'], format = "%d.%m.%Y")

#transactions['dt_year'] = transactions['date'].dt.year

#transactions['dt_month'] = transactions['date'].dt.month

#transactions['dt_day'] = transactions['date'].dt.day

#transactions.head(5)

#transactions.groupby(['dt_year', 'dt_month'])['dt_day'].nunique()



# 'date_block_column'

#transactions.groupby(['dt_year', 'dt_month'])['date_block_num'].value_counts()
# matrix do dlarionov é o mesmo que o meu grid.

matrix = []

cols = ['date_block_num','shop_id','item_id']

for i in range(34):

    sales = sales_train[sales_train.date_block_num==i]

    matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))

    

matrix = pd.DataFrame(np.vstack(matrix), columns=cols)

matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)

matrix['shop_id'] = matrix['shop_id'].astype(np.int8)

matrix['item_id'] = matrix['item_id'].astype(np.int16)

matrix.sort_values(cols,inplace=True)
# 'shop_item_cnt_month'

group = sales_train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})

group.columns = ['shop_item_cnt_mday']

group.reset_index(inplace=True)



matrix = pd.merge(matrix, group, on=cols, how='left')

matrix['shop_item_cnt_mday'] = (matrix['shop_item_cnt_mday']

                                .fillna(0)

                                .clip(0,20)

                                .astype(np.float16))

# .clip(0,20) # NB clip target here -> entre fillna e astype
matrix['month'] = matrix['date_block_num'] % 12



days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])

matrix['days'] = matrix['month'].map(days).astype(np.int8)



matrix['shop_item_cnt_mday'] = matrix['shop_item_cnt_mday'] / matrix['days']

matrix = matrix.drop(columns = ['month', 'days'])
test_in_train = test_df.copy()

test_in_train['date_block_num'] = 34

test_in_train['date_block_num'] = test_in_train['date_block_num'].astype(np.int8)

test_in_train['shop_id'] = test_in_train['shop_id'].astype(np.int8)

test_in_train['item_id'] = test_in_train['item_id'].astype(np.int16)
matrix = pd.concat([matrix, test_in_train], ignore_index=True, sort=False, keys=cols)

matrix.fillna(0, inplace=True) # 34 month
matrix = matrix.join(items.set_index('item_id'),on = 'item_id')

matrix = matrix.join(shops.set_index('shop_id'), on = 'shop_id')

matrix.head()
# Quando analisamos os números totais, vemos uma série temporal com sazonalidade e tendência de queda.



vis_total = sales_train.groupby(['date_block_num'])[['item_cnt_day', 'shop_id']].sum()

vis_total = vis_total.drop(columns = 'shop_id')

sns.lineplot(x = vis_total.index, y = 'item_cnt_day', data = vis_total)
# Porém, temos que estimar um número 'capped' entre [0,20] e a 'série' fica estável e estacionária.

# A queda no fim é pq o test_set tem vendas 0 (a serem estimadas)



#vis_total = matrix.groupby(['date_block_num'])[['shop_item_cnt_mday', 'shop_id']].sum()

#vis_total = vis_total.drop(columns = 'shop_id')

#sns.lineplot(x = vis_total.index, y = 'shop_item_cnt_mday', data = vis_total)
# Quando plotamos o 'item_id' vemos que tem muitos produtos novos que surgem e outros que saem

# começa com uma venda alta e cai ao longo do tempo.

# Como são ~20K produtos, o plot demora para rodar. Abaixo está o código.



#vis_item_id = vis.groupby(['date_block_num', 'item_id'], as_index = False)['target'].sum()

#sns.lineplot(x = 'date_block_num', y = 'target', hue = 'item_id', data = vis_item_id)
def lag_feature(df, lags, col):

    tmp = df[['date_block_num','shop_id','item_id',col]]

    for i in lags:

        shifted = tmp.copy()

        shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]

        shifted['date_block_num'] += i

        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')

    return df
matrix = lag_feature(matrix, [1,2,3,6,12], 'shop_item_cnt_mday')
import time

ts = time.time()
# Estava fazendo 'shop_cnt_month' e 'item_cnt_month' por 'sum'. Porém, esses valores podem estar

# 'distorcidos' por items com muita venda e não representar a média de todos.



ts = time.time()



# Month

cols = ['date_block_num']

name_col = 'm_cnt_avg'

group = matrix.groupby(cols).agg({'shop_item_cnt_mday': ['mean']})

group.columns = [name_col]

group.reset_index(inplace=True)



matrix = pd.merge(matrix, group, on= cols, how='left')

matrix[name_col] = (matrix[name_col].fillna(0).astype(np.float16))



matrix = lag_feature(matrix, [1,2,3,6,12], name_col)

matrix.drop(columns = [name_col], axis=1, inplace=True)



# Month-ShopId

cols = ['date_block_num', 'shop_id']

name_col = 'm_shop_cnt_avg'

group = matrix.groupby(cols).agg({'shop_item_cnt_mday': ['mean']})

group.columns = [name_col]

group.reset_index(inplace=True)



matrix = pd.merge(matrix, group, on= cols, how='left')

matrix[name_col] = (matrix[name_col].fillna(0).astype(np.float16))



matrix = lag_feature(matrix, [1,2,3,6,12], name_col)

matrix.drop(columns = [name_col], axis=1, inplace=True)



# Month - ItemId

cols = ['date_block_num', 'item_id']

name_col = 'm_item_cnt_avg'

group = matrix.groupby(cols).agg({'shop_item_cnt_mday': ['mean']})

group.columns = [name_col]

group.reset_index(inplace=True)



matrix = pd.merge(matrix, group, on= cols, how='left')

matrix[name_col] = (matrix[name_col].fillna(0).astype(np.float16))



matrix = lag_feature(matrix, [1,2,3,6,12], name_col)

matrix.drop(columns = [name_col], axis=1, inplace=True)



time.time() - ts
ts = time.time()



# Month - Category

cols = ['date_block_num', 'item_category_id']

name_col = 'm_categ_cnt_avg'

group = matrix.groupby(cols).agg({'shop_item_cnt_mday': ['mean']})

group.columns = [name_col]

group.reset_index(inplace=True)



matrix = pd.merge(matrix, group, on= cols, how='left')

matrix[name_col] = (matrix[name_col].fillna(0).astype(np.float16))



matrix = lag_feature(matrix, [1,2,3,6,12], name_col)

matrix.drop(columns = [name_col], axis=1, inplace=True)



# Month - MacroCategory

cols = ['date_block_num', 'item_macro_categ_code']

name_col = 'm_macrocateg_cnt_avg'

group = matrix.groupby(cols).agg({'shop_item_cnt_mday': ['mean']})

group.columns = [name_col]

group.reset_index(inplace=True)



matrix = pd.merge(matrix, group, on= cols, how='left')

matrix[name_col] = (matrix[name_col].fillna(0).astype(np.float16))



matrix = lag_feature(matrix, [1,2,3,6,12], name_col)

matrix.drop(columns = [name_col], axis=1, inplace=True)



# Month - SubCategory

cols = ['date_block_num', 'item_subcateg_code']

name_col = 'm_subcateg_cnt_avg'

group = matrix.groupby(cols).agg({'shop_item_cnt_mday': ['mean']})

group.columns = [name_col]

group.reset_index(inplace=True)



matrix = pd.merge(matrix, group, on= cols, how='left')

matrix[name_col] = (matrix[name_col].fillna(0).astype(np.float16))



matrix = lag_feature(matrix, [1,2,3,6,12], name_col)

matrix.drop(columns = [name_col], axis=1, inplace=True)



# Month - Shop - Category

cols = ['date_block_num', 'shop_id', 'item_category_id']

name_col = 'm_shop_categ_cnt_avg'

group = matrix.groupby(cols).agg({'shop_item_cnt_mday': ['mean']})

group.columns = [name_col]

group.reset_index(inplace=True)



matrix = pd.merge(matrix, group, on= cols, how='left')

matrix[name_col] = (matrix[name_col].fillna(0).astype(np.float16))



matrix = lag_feature(matrix, [1,2,3,6,12], name_col)

matrix.drop(columns = [name_col], axis=1, inplace=True)



# Month - Shop - MacroCategory

cols = ['date_block_num', 'shop_id', 'item_macro_categ_code']

name_col = 'm_shop_macrocateg_cnt_avg'

group = matrix.groupby(cols).agg({'shop_item_cnt_mday': ['mean']})

group.columns = [name_col]

group.reset_index(inplace=True)



matrix = pd.merge(matrix, group, on= cols, how='left')

matrix[name_col] = (matrix[name_col].fillna(0).astype(np.float16))



matrix = lag_feature(matrix, [1,2,3,6,12], name_col)

matrix.drop(columns = [name_col], axis=1, inplace=True)



# Month - Shop - SubCategory

cols = ['date_block_num', 'shop_id', 'item_subcateg_code']

name_col = 'm_shop_subcateg_cnt_avg'

group = matrix.groupby(cols).agg({'shop_item_cnt_mday': ['mean']})

group.columns = [name_col]

group.reset_index(inplace=True)



matrix = pd.merge(matrix, group, on= cols, how='left')

matrix[name_col] = (matrix[name_col].fillna(0).astype(np.float16))



matrix = lag_feature(matrix, [1,2,3,6,12], name_col)

matrix.drop(columns = [name_col], axis=1, inplace=True)



time.time() - ts
ts = time.time()



# Month - City

cols = ['date_block_num', 'city_code']

name_col = 'm_city_cnt_avg'

group = matrix.groupby(cols).agg({'shop_item_cnt_mday': ['mean']})

group.columns = [name_col]

group.reset_index(inplace=True)



matrix = pd.merge(matrix, group, on= cols, how='left')

matrix[name_col] = (matrix[name_col].fillna(0).astype(np.float16))



matrix = lag_feature(matrix, [1,2,3,6,12], name_col)

matrix.drop(columns = [name_col], axis=1, inplace=True)



# Month - StoreType

cols = ['date_block_num', 'store_type']

name_col = 'm_storetype_cnt_avg'

group = matrix.groupby(cols).agg({'shop_item_cnt_mday': ['mean']})

group.columns = [name_col]

group.reset_index(inplace=True)



matrix = pd.merge(matrix, group, on= cols, how='left')

matrix[name_col] = (matrix[name_col].fillna(0).astype(np.float16))



matrix = lag_feature(matrix, [1,2,3,6,12], name_col)

matrix.drop(columns = [name_col], axis=1, inplace=True)



time.time() - ts
# Variação de preço MoM



group = sales_train.groupby(['date_block_num', 'shop_id','item_id']).agg({'item_price': ['mean']})

group.columns = ['m_item_price']

group.reset_index(inplace=True)



matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id','item_id'], how='left')

matrix['m_item_price'] = matrix['m_item_price'].astype(np.float32)



matrix = lag_feature(matrix, [1], 'm_item_price')



matrix['delta_pc_price_mom'] = (matrix['m_item_price'] - matrix['m_item_price_lag_1'])/ matrix['m_item_price_lag_1']



matrix.drop(['m_item_price', 'm_item_price_lag_1'], axis=1, inplace=True)
# Months since first sale



matrix['item_shop_first_sale'] = matrix['date_block_num'] - matrix.groupby(['item_id','shop_id'])['date_block_num'].transform('min')

matrix['item_first_sale'] = matrix['date_block_num'] - matrix.groupby('item_id')['date_block_num'].transform('min')
matrix = matrix[matrix['date_block_num'] > 11]
# Filling lag na's



def fill_na(df):

    for col in df.columns:

        if ('_lag_' in col) & (df[col].isnull().any()):

            if ('item_cnt' in col):

                df[col].fillna(0, inplace=True)         

    return df



matrix = fill_na(matrix)

matrix['delta_pc_price_mom'].fillna(0, inplace = True)

matrix['item_shop_first_sale'].fillna(0, inplace = True)

matrix['item_first_sale'].fillna(0, inplace = True)
matrix.columns
del sales_train

del item_cat

del items

del shops
corr_data = matrix.loc[matrix['date_block_num'] < 34]



#correlation heatmap of dataset

def correlation_heatmap(df):

    _ , ax = plt.subplots(figsize =(14, 12))

    colormap = sns.diverging_palette(220, 10, as_cmap = True)

    

    _ = sns.heatmap(

        df.corr(), 

        cmap = colormap,

        square=True, 

        cbar_kws={'shrink':.9 }, 

        ax=ax,

        annot=True, 

        linewidths=0.1,vmax=1.0, linecolor='white',

        annot_kws={'fontsize':12 }

    )

    

    plt.title('Pearson Correlation of Features', y=1.05, size=15)



correlation_heatmap(corr_data)
corr = corr_data.corr()
corr_target = pd.DataFrame(corr['shop_item_cnt_mday'])

corr_target_abs = corr_target.apply(abs)

corr_target_abs_as = corr_target_abs.sort_values(by = ['shop_item_cnt_mday'], ascending = True)
matrix.shape
corr_target_abs_as.head(42)
matrix = matrix.drop(columns = ['m_cnt_avg_lag_3',

                                'm_cnt_avg_lag_6',

                                'm_cnt_avg_lag_2',

                                'm_cnt_avg_lag_1',

                                'm_cnt_avg_lag_12',

                                'm_storetype_cnt_avg_lag_6',

                                'm_storetype_cnt_avg_lag_3',

                                'm_storetype_cnt_avg_lag_2',

                                'm_storetype_cnt_avg_lag_1',

                                'm_storetype_cnt_avg_lag_12',

                                'm_city_cnt_avg_lag_6',

                                'm_city_cnt_avg_lag_12',

                                'm_city_cnt_avg_lag_3',

                                'm_city_cnt_avg_lag_2',

                                'm_city_cnt_avg_lag_1',

                                'm_shop_cnt_avg_lag_6',

                                'm_shop_cnt_avg_lag_12',

                                'm_shop_cnt_avg_lag_3',

                                'm_item_cnt_avg_lag_12',

                                'm_shop_cnt_avg_lag_2',

                                'm_shop_cnt_avg_lag_1',

                                'm_macrocateg_cnt_avg_lag_1',

                                'm_macrocateg_cnt_avg_lag_6',

                                'm_macrocateg_cnt_avg_lag_12',

                                'shop_item_cnt_mday_lag_12',

                                'm_macrocateg_cnt_avg_lag_3',

                                'm_macrocateg_cnt_avg_lag_2',

                                'm_item_cnt_avg_lag_6',

                                'shop_item_cnt_mday_lag_6'])
print(matrix.shape)

matrix.columns
matrix.to_pickle('data.pkl')

del matrix

del group

# leave test for submission

gc.collect();
data = pd.read_pickle('data.pkl')
# Validation strategy is 34 month for the test set, 33 month for the validation set and 13-33 months for the train.



X_train = data[data.date_block_num < 33].drop(['shop_item_cnt_mday'], axis=1)

Y_train = data[data.date_block_num < 33]['shop_item_cnt_mday']

X_valid = data[data.date_block_num == 33].drop(['shop_item_cnt_mday'], axis=1)

Y_valid = data[data.date_block_num == 33]['shop_item_cnt_mday']

X_test = data[data.date_block_num == 34].drop(['shop_item_cnt_mday'], axis=1)
del data

gc.collect();
model = XGBRegressor(

    max_depth=8,

    n_estimators=200,

    min_child_weight=300, 

    colsample_bytree=0.8, 

    subsample=0.8, 

    eta=0.3,    

    seed=42)



model.fit(

    X_train, 

    Y_train, 

    eval_metric="rmse", 

    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 

    verbose=True, 

    early_stopping_rounds = 5)
Y_pred = model.predict(X_valid).clip(0, 20)

Y_test1 = model.predict(X_test)

Y_test2 = Y_test1 * 31

Y_test3 = Y_test2.clip(0, 20)



submission = pd.DataFrame({

    "ID": test_df.index, 

    "item_cnt_month": Y_test3

})

submission.to_csv('xgb_submission.csv', index=False)
plot_features(model, (10,14))
# TBD