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
# plot daily sales average

data_means = pd.DataFrame(data.mean(), columns=['mean'])

data_means.plot(subplots=True, figsize=(15,4))

# Strip holidays from calendar and create unique list

holidays = calendar[['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']]

uholidays = pd.unique(holidays[['event_name_1', 'event_name_2']].values.ravel())[1:]

holidays_shifted = holidays.shift(-14).loc[:data_means.shape[0]-1, :]

uholidays

# align dates to each holiday

for holiday in uholidays:

    dayno = 0

    daynos = []

    for index, row in holidays_shifted.iterrows():

        if dayno > 0:

            dayno += 1

        if dayno>21:

            dayno = 0

        if (row.event_name_1==holiday) | (row.event_name_2==holiday): # upcoming holiday

            dayno = 1

        daynos.append(dayno)



    data_means['dayno'] = daynos

    data_means['dayno'] -= 15

    df = data_means.groupby('dayno', sort=True).mean() #.plot(figsize=(15,2))

    df.columns = [holiday]

    df['ref'] = df.iloc[0, 0] # all zero dayno's go here

    ax = df[1:].plot(figsize=(15,4))

    ax.locator_params(integer=True)

    ax.axvline(x=0)

    plt.show()
