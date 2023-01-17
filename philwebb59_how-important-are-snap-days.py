# import libraries

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

datapath = '../input/m5-forecasting-accuracy'



# import data files

calendar = pd.read_csv(f'{datapath}/calendar.csv', parse_dates=['date'])

sales_train_validation = pd.read_csv(f'{datapath}/sales_train_validation.csv')
# tags = 'id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'

# data = 'd_1' -> 'd_1913'

tags, data =sales_train_validation.iloc[:, :6], sales_train_validation.iloc[:, 6:]
# add a key column merging state & category

tags['key'] = tags[['state_id','cat_id']].agg('_'.join, axis=1)

# plot daily sales grouped by key

data_means = data.groupby(tags.key).mean().transpose()

data_means.plot(subplots=True, figsize=(15,4*data_means.shape[1]))

# strip state from each key and list unique states

states = data_means.columns.str[0:2]

ustates = states.unique()

# compare SNAP days to non-SNAP days

summary = []

for state in ustates:

    mask = (states==state).tolist()

    df = pd.DataFrame(data_means.iloc[:, mask])

    df['snap'] = calendar.loc[:df.shape[0]-1,f'snap_{state}'].tolist()

    summary.append(df.groupby(['snap']).mean())



summary = pd.concat(summary, axis=1).rename(index={0:'non_SNAP',1:'SNAP'}).transpose()

summary['% Change'] = summary['SNAP'] / summary['non_SNAP'] - 1

print(summary)