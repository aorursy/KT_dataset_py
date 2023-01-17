# imports

import pandas as pd

import numpy as np
#seed the randomizer

np.random.seed(1)



# define number of records to create

record_cnt = 10000



# Define data choice lists

origin = ['Cuba', 'Dominican Republic', 'Nicaragua', 'Honduras', 'Ecuador', 'Mexico', 'Brazil', 'USA', 'Jamaica', 'Cameroon']

destin = ['USA', 'Belgium', 'Germany', 'Malaysia', 'Netherlands']

mode = ['Land', 'Sea', 'Air']

inspect = [0, 1]



# create the base data frame and populate with base random data

df_base = pd.DataFrame({'origin': np.random.choice(origin, record_cnt, p=[0.25, 0.1, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05, 0.025, 0.025]),

                       'destin': np.random.choice(destin, record_cnt, p=[0.25, 0.2, 0.2, 0.2, 0.15]),

                       'mode': np.random.choice(mode, record_cnt, p=[0.10, 0.55, 0.35]),

                       'qty':  np.random.randint(low=100, high=150, size=record_cnt),

                       'value': np.random.randint(low=2500, high=6000, size=record_cnt),

                        'importer_id': np.random.randint(low=1000, high=1005, size=record_cnt),

                        'exporter_id': np.random.randint(low=2000, high=2009, size=record_cnt),

                       'inspected': np.random.choice(inspect, record_cnt, p=[0.91, 0.09])})

df_base.head()
import seaborn as sns

import matplotlib.pyplot as plt



df_base.hist(bins=20, figsize=(20,10))

plt.show()
plt.figure(figsize=(20,5))

sns.boxplot(x='origin', y='qty', data=df_base)

plt.show()
plt.figure(figsize=(20,5))

sns.boxplot(x='destin', y='qty', data=df_base)

plt.show()
df_base['weight'] = (df_base['value'] / 10)

df_base.head()
df_base['days_in_transit'] = 0    # create the new column and set all rows to 0



df_base.head()
# loop through dataframe and change the days in transit based on mode {'Land':2-7, 'Sea':7-21, 'Air':1-3}

for i, row in df_base.iterrows():

    if df_base.loc[i, 'mode'] == 'Land': 

        df_base.at[i, 'days_in_transit'] = (np.random.randint(low=2, high=7))

    elif df_base.loc[i,'mode'] == 'Sea': 

        df_base.at[i, 'days_in_transit'] = (np.random.randint(low=7, high=21))

    else: 

        df_base.at[i, 'days_in_transit'] = (np.random.randint(low=1, high=3))

df_base.head()

    
df_base.shape
df_base.to_csv('shipping.csv')