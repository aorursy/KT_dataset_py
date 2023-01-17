import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
def open_csv(file):
    path = '/kaggle/input/competitive-data-science-predict-future-sales'
    return pd.read_csv(os.path.join(path, file))

# Abre os arquivos
items = open_csv('items.csv')
shops = open_csv('shops.csv')
item_categories = open_csv('item_categories.csv')
train = open_csv('sales_train.csv')
submission = open_csv('sample_submission.csv')
test = open_csv('test.csv')
print(f'Número de elementos no dataframe de treino: {train.shape[0]}')
print(f'Quantidade de lojas diferentes: {shops.shape[0]}')
print(f'Quantidade de itens diferentes: {items.shape[0]}')
print(f'Quantidade de categorias diferentes: {item_categories.shape[0]}')
print(train.columns)
# Remove dados duplicados
attr = ['date', 'date_block_num', 'shop_id', 'item_id', 'item_cnt_day']
old_size = train.shape[0]
train.drop_duplicates(attr, inplace=True)
print(f'Quantidade de elementos duplicados: {old_size - train.shape[0]}')
# Retira dados sem relevância para a análise
train.drop(columns=['date'], inplace=True)
items.drop(columns=['item_name'], inplace=True)
test.drop(columns=['ID'], inplace=True)
# Retira outliers
old_size = train.shape[0]
print('Vendas:')
print(train['item_cnt_day'].describe())
train.boxplot(column='item_cnt_day')
plt.show()
print('\nPreço:')
print(train['item_price'].describe())
train.boxplot(column='item_price')
plt.show()
train = train[(train['item_cnt_day'] >= 0) &
              (train['item_cnt_day'] <= 800) &
              (train['item_price'] >= 0) &
              (train['item_price'] <= 100000)]

print(f'Quantidade de elementos retirados: {old_size - train.shape[0]}')
# Adiciona a categoria dos itens ao dataframe
train = pd.merge(train, items, on='item_id', how='left')
test = pd.merge(test, items, on='item_id', how='left')
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memória utilizada pelo dataframe: {:.2f} MB.'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memória utilizada após a otimização: {:.2f} MB.'.format(end_mem))
    print('Uso de memória reduzido em {:.1f}%.'.format(100 * (start_mem - end_mem) / start_mem))

    return df
# Reduz a memória utilizada
train = reduce_mem_usage(train)
from itertools import product

index_cols = ['shop_id', 'item_id', 'date_block_num']
# Cria uma combinação de cada loja/item daquele mês.
grid = []
for block_num in train['date_block_num'].unique():
    cur_shops = train.loc[train['date_block_num'] == block_num, 'shop_id'].unique()
    cur_items = train.loc[train['date_block_num'] == block_num, 'item_id'].unique()
    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])), dtype='int32'))
grid = pd.DataFrame(np.vstack(grid), columns=index_cols, dtype=np.int32)

print(f'\nQuantidade de elementos: {grid.shape[0]}\n')

print(grid)
# Reduz a memória utilizada
grid = reduce_mem_usage(grid)
sales_m = train.groupby(index_cols).agg({'item_cnt_day': 'sum',
                                         'item_price': np.mean}).reset_index()

print(sales_m)
sales_m = pd.merge(grid, sales_m, on=index_cols, how='left').fillna(0)

print(f'Quantidade de elementos: {sales_m.shape[0]}\n')
print(sales_m)
sales_m = pd.merge(sales_m, items, on='item_id', how='left')
for type_id in ['item_id', 'shop_id', 'item_category_id']:
    for column_id, aggregator, aggtype in [('item_price', np.mean, 'avg'),
                                           ('item_cnt_day', np.sum, 'sum'),
                                           ('item_cnt_day', np.mean, 'avg')]:

        # Gera os novos atributos e renomeia as colunas
        mean_df = train.groupby([type_id, 'date_block_num']).aggregate(aggregator).reset_index()[[column_id, type_id, 'date_block_num']]
        mean_df.columns = [type_id+'_'+aggtype+'_'+column_id, type_id, 'date_block_num']
        
        # Une os novos atributos ao dataframe original
        sales_m = pd.merge(sales_m, mean_df, on=['date_block_num', type_id], how='left')
    
print(sales_m.columns)
# Definimos quais variáveis serão analisadas dos meses passados
lag_variables = list(sales_m.columns[6:])+['item_cnt_day']
# Definimos a janela temporal
lags = [1, 2, 3, 4, 5, 6, 12]
    
def lag_features(df, sales_m):
    for lag in lags:
        # Cria-se uma cópia do sales_m
        sales_new_df = sales_m.copy()
        # Translada os meses adicionando o valor lag a eles
        sales_new_df.date_block_num += lag
        # Adiciona as novas colunas com as variáveis temporais
        sales_new_df = sales_new_df[index_cols+lag_variables]
        sales_new_df.columns = index_cols + [f'{lag_feat}_lag_{lag}'
                                             for lag_feat in lag_variables]
        # Une o dataset original com o com os atributos temporais, usando de referência
        # o identificador do item e da loja, e o mês de venda (index_cols).
        df = pd.merge(df, sales_new_df, on=index_cols, how='left')

    return df


lag_train = sales_m.copy()
lag_train = lag_features(lag_train, sales_m)

print(f'Quantidade de atributos no dataframe: {lag_train.shape[1]}\n')
print(lag_train.columns)
def fillNaN(df):
    for feat in df.columns:
        if 'item_cnt' in feat:
            df[feat] = df[feat].fillna(0)
        elif 'item_price' in feat:
            df[feat] = df[feat].fillna(df[feat].median())

fillNaN(lag_train)
def date_attr(df):
    df['month'] = df['date_block_num'] % 12
    df['year'] = df['date_block_num'] // 12
    df.drop(columns='date_block_num', inplace=True)
    return df

# Retira os meses do primeiro ano
lag_train = lag_train[lag_train['date_block_num'] > 12]
# Gera novos atributos de ano e mês
lag_train = date_attr(lag_train)
# Retira os atributos que não serão utilizados na análise,
# como os atributos utilizados para gerar as lag features.
cols_to_drop = lag_variables[:-1] + ['item_price']
lag_train.drop(columns=cols_to_drop, inplace=True)

# Reduz a memória utilizada
lag_train = reduce_mem_usage(lag_train)
test['date_block_num'] = 34

test = lag_features(test, sales_m)
fillNaN(test)
test = date_attr(test)
# Verificando se o treino e o teste possuem as mesmas colunas
_test = set(test.columns)
_train = set(lag_train.drop(columns='item_cnt_day').columns)
assert _test==_train
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=10, n_jobs = -1)

regressor.fit(lag_train.drop(columns=['item_cnt_day']),
              lag_train['item_cnt_day'])

pred = regressor.predict(test)

pred = pred.clip(0, 20)


submission['item_cnt_month'] = pred

submission.to_csv('submission_rf.csv', index=False)
# Feature importance
feat_importances = pd.Series(regressor.feature_importances_, index=lag_train.drop(columns=['item_cnt_day']).columns)
feat_importances.nlargest(10).plot(kind='barh')
param = {'max_depth': 10,
         'subsample': 1,
         'min_child_weight': 1,
         'eta': 0.3,
         'num_round': 1000,
         'eval_metric': 'rmse',
         'verbosity': 0}
import xgboost as xgb

xgbtrain = xgb.DMatrix(lag_train.drop(columns=['item_cnt_day']),
                       lag_train['item_cnt_day'])

bst = xgb.train(param, xgbtrain)

xgbpredict = xgb.DMatrix(test)
pred = bst.predict(xgbpredict)
pred = pred.clip(0, 20)

submission['item_cnt_month'] = pred

submission.to_csv('submission_xgboost.csv', index=False)
# Feature importance
x = xgb.plot_importance(bst)
x.figure.set_size_inches(10, 30)