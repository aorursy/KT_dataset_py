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
cats = sales_train_validation['cat_id'].unique().tolist()

print(cats)
item_means = data.mean(axis=1)
scaled_data = data.div(item_means.iloc[0], axis='columns')
# plot daily sales average

data_means = {}

for cat in cats:

    data_means[cat] = pd.DataFrame(scaled_data[tags['cat_id']==cat].mean(), columns=['mean'])

    data_means[cat].plot(subplots=True, figsize=(15,4), title=cat)
# Strip holidays from calendar and create unique list

holidays = calendar[['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']]

uholidays = pd.unique(holidays[['event_name_1', 'event_name_2']].values.ravel())[1:]

holidays_shifted = holidays.shift(-14).loc[:data.shape[1]-1, :]

uholidays
# align dates to each holiday

for holiday in uholidays:

    dayno = 0

    daynos = []

    for index, row in holidays_shifted.iterrows():

        if dayno > 0:

            dayno += 1

        if dayno > 21:

            dayno = 0

        if (row.event_name_1 == holiday) or (row.event_name_2 == holiday): # upcoming holiday

            dayno = 1

        daynos.append(dayno)



    for cat, means in data_means.items():

        means['dayno'] = daynos

        means['dayno'] -= 15

        df = means.groupby('dayno', sort=True).mean() #.plot(figsize=(15,2))

        df.columns = [f'{holiday} ({cat})']

        df[f'ref ({cat})'] = df.iloc[0, 0] # all zero dayno's go here

        ax = df[1:].plot(figsize=(15,4))

        ax.locator_params(integer=True)

        ax.axvline(x=0)

    plt.show()