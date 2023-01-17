import pandas as pd
import numpy as np
chipo = pd.read_csv('../input/chipotle.tsv', sep = '\t')
chipo

chipo[:10]
chipo.head(10)
chipo[-10:]
chipo.tail(10)

#наблюдение - строки, признаки - столбцы 
print(chipo.shape)
print(chipo.shape[0])
print(chipo.shape[1])

chipo.columns

chipo.index
chipo['item_name'].value_counts().index[0]
chipo['item_name'].value_counts()[0]
chipo['item_name'].value_counts().shape[0]
chipo['item_price']
def dollar(s):
    return float(s[1:])
chipo['item_price'] = chipo['item_price'].map(dollar)

chipo[:5]
chipo['item_price']
np.sum(chipo['item_price'])
chipo['order_id'].value_counts().size
a = np.sum(chipo['item_price'])
b = chipo['order_id'].value_counts().size
print(a/b)
a = chipo[['item_name', 'item_price']]
a
a = a.sort_values('item_name')
a
name = chipo[chipo['quantity']==1]
sum(chipo[chipo['item_name']==name[name['item_price']==max(name['item_price'])].iloc[0,2]]['quantity'])
sum(chipo[chipo['item_name']=='Veggie Salad Bowl']['quantity'])
len(chipo[chipo['item_name']=='Canned Soda'][chipo[chipo['item_name']=='Canned Soda']>1])