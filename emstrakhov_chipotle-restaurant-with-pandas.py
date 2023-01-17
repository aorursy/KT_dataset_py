import numpy as np

import pandas as pd
chipo = pd.read_csv('../input/chipotle.tsv', 

                    sep='\t') # разделитель - табуляция
chipo.head(10)
print('Строк:', chipo.shape[0])

print('Столбцов:', chipo.shape[1])
print(chipo.columns) # колонки как индекс
print(chipo.columns.values) # сами названия
print(chipo.index)
chipo['item_name'].value_counts().index[0] # value_counts сортирует значения от больших к меньшим
chipo['item_name'].value_counts().head(1) # вся строка
chipo['item_name'].value_counts()[0] # выбрали только число
chipo['item_name'].value_counts().shape[0]
type(chipo['item_price']) # тип самой колонки (не значений в ней)
chipo['item_price'].dtype # тип значений
def _(x):

    return float(x.strip()[1:]) # убрали лишние пробелы и первый символ, перевели в float



chipo['item_price_cleaned'] = chipo['item_price'].apply(_)

chipo.head()
del chipo['item_price']

chipo.head()
chipo['item_price_cleaned'].dtype
# Вариант 1 - без учёта quantity

print(chipo['item_price_cleaned'].sum()) # !

print(np.sum(chipo['item_price_cleaned']))

print(sum(chipo['item_price_cleaned']))
# Вариант 2 - с учётом quantity

print((chipo['quantity']*chipo['item_price_cleaned']).sum())
chipo['order_id'].unique().shape[0] # unique() оставляет только уникальные значения
len(chipo['order_id'].unique()) # альтернативный способ
chipo['price'] = chipo['quantity']*chipo['item_price_cleaned']

chipo.head()
# Группировка по номеру заказа + сумма по price, затем среднее

print('Средний чек:', np.round(chipo.groupby('order_id')['price'].sum().mean(), 2))
items = chipo[['item_name', 'item_price_cleaned']]

items.head()
items_sorted = items.sort_values(by='item_name') # по умолчанию - сортировка по возрастанию

items_sorted.head(10)
items['price'] = chipo['item_price_cleaned'] / chipo['quantity']

items.head()
items_sorted = items.sort_values(by='price', ascending=False) # по убыванию; самое дорогое блюдо вверху

items_sorted.head()
most_expensive = items_sorted.iloc[0, 0]

print(most_expensive)
# Фильтр по названию блюда + сумма по столбцу quantity

print('Кол-во заказов самого дорогого:',

      chipo[ chipo['item_name']==most_expensive ]['quantity'].sum())
print('Кол-во заказов Veggie Salad Bowl:',

      chipo[ chipo['item_name']=='Veggie Salad Bowl' ]['quantity'].sum())
# Фильтр одновременно по двум условиям

chipo[ (chipo['item_name']=='Canned Soda') & (chipo['quantity']>1) ].shape[0]
chipo[ (chipo['item_name']=='Canned Soda') & (chipo['quantity']>1) ]