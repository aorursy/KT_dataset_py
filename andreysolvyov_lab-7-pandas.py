import pandas as pd
import numpy as np
chipo = pd.read_csv('../input/chipotle.tsv', sep = '\t')
chipo.head(10)
chipo.iloc[-10:, :]
chipo.tail(10)
print(len(chipo.index), '- кол-во наблюдений')
print(len(chipo.columns), '- кол-во признаков')
print(chipo.columns)
chipo.index
chipo['item_name'].value_counts().index[0]
chipo['item_name'].value_counts()[0]
len(chipo['item_name'].value_counts())
def f(s):
    return float(s[1:])
print(chipo['item_price'].dtype)
chipo['item_price'] = chipo['item_price'].map(lambda s: float(s[1:]))
chipo['item_price']
np.sum(chipo['item_price'])
len(set(chipo['order_id']))
np.sum(chipo['item_price'])/len(set(chipo['order_id']))
items = chipo[['item_name', 'item_price']]
items
items = items.sort_values('item_name')
items
name = chipo[chipo['quantity'] == 1]
sum(chipo[chipo['item_name'] == name[name['item_price'] == max(name['item_price'])].iloc[0, 2]]['quantity'])

sum(chipo[chipo['item_name'] == 'Veggie Salad Bowl']['quantity'])
len(chipo[chipo['item_name'] == 'Canned Soda'][chipo[chipo['item_name'] == 'Canned Soda'] > 1])
