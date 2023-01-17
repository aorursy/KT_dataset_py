import numpy as np

import pandas as pd



data = pd.read_csv('../input/train.csv',index_col='id',parse_dates=['timestamp'])



#we have to drop entries with unlikely low full_sq

print(len(data[data['full_sq']<15]), ' entries drop')

data = data.drop((data[data['full_sq']<15]).index)



data['price_per_meter'] = data['price_doc']/data['full_sq']
data.loc[:,data.columns.drop('price_per_meter')].corr()['price_doc'].sort_values()
data.loc[:,data.columns.drop('price_doc')].corr()['price_per_meter'].sort_values()