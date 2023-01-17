import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import glob, re

from datetime import datetime
dfs = { re.search('/([^/\.]*)\.csv', fn).group(1):pd.read_csv(fn) for fn in glob.glob('../input/*.csv')}

print('data frames read:{}'.format(list(dfs.keys())))



print('local variables with the same names are created.')

for k, v in dfs.items(): locals()[k] = v
store_list = np.unique(air_visit_data['air_store_id'])

for cnt, store_id in enumerate(store_list):   

    data = air_visit_data.loc[air_visit_data.air_store_id == store_id]

    data['visit_date'] = pd.to_datetime(data['visit_date'])

    store_data = pd.DataFrame()

    store_data['visit_date'] = pd.date_range('20160101','20170531')

    store_data = store_data.merge(data, on='visit_date', how='left')

    store_data.index = store_data['visit_date']

    store_data['visitors'].interpolate(method='time', inplace=True)

    store_data.drop(['visit_date', 'air_store_id'], axis=1, inplace=True)

    #store_data



    plt.figure(figsize=(20,5))

    plt.plot(store_data)

    plt.title(store_id)

    plt.xlim(store_data.index.min(), store_data.index.max())

    plt.grid()

    plt.show()

    

    if(cnt>10): break