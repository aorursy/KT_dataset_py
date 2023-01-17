import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

plt.style.use('fivethirtyeight')

drop_data= pd.read_csv('../input/indian-school-education-statistics/dropout-ratio-2012-2015.csv')

drop_data.columns
drop_data= pd.read_csv('../input/indian-school-education-statistics/dropout-ratio-2012-2015.csv',parse_dates=['year'],index_col='year')

drop_data.columns=['state',

                   '1_4_boys',

                   '1_4_girls',

                   '1_4_total',

                   '5_8_boys',

                   '5_8_girls',

                   '5_8_total',

                   '9_10_boys',

                   '9_10_girls',

                   '9_10_total',

                   '11_12_boys',

                   '11_12_girls',

                   '11_12_total'

                  ]
drop_data.head(20)
import re

col= ['1_4_boys','1_4_girls','1_4_total','5_8_boys','5_8_girls','5_8_total','9_10_boys','9_10_girls','9_10_total','11_12_boys','11_12_girls','11_12_total']

drop_data[col]= drop_data[col].replace('NR',np.nan)

drop_data[col]= drop_data[col].replace('Uppe_r_Primary',np.nan).astype(float)

drop_data
drop_data.isna().sum()
states={}

for state_name in drop_data['state'].unique():

    state = drop_data.query(f'state== "{state_name}"').copy()

    col= state.isna().sum()[state.isna().sum()>0].index

    state[col]= state[col].fillna(state[col].mean())

    states[str(state_name)]= state



print(states.keys())
# means={}

state='A & N Islands'

AN_island = drop_data.query(f'state== "{state}"').copy()

print('-'*50,'With empty values','-'*50)

display(AN_island)

print('-'*50,'Without empty values','-'*50)

AN_island.fillna(method='bfill',inplace=True)

display(AN_island)

col=[ '1_4_boys',

      '1_4_girls', 

      '5_8_boys', 

      '5_8_girls',

      '9_10_boys', 

      '9_10_girls', 

      '11_12_boys',

      '11_12_girls',

    ]

print('-'*50,'With importantn columns','-'*50)

display(AN_island[col])

print('-'*50,'Bar Graph of above columns','-'*50)

AN_island[col].plot(kind='bar',figsize=(12,5))

plt.xticks(np.arange(3),['2012-13','2013-14','2014-15'])

plt.show()

means={}

for state_name in states.keys():

    state = states[state_name]

    state['mean']= state.mean(axis=1)

    means[str(state_name)]= state['mean'].sum()
for state in states.keys():

    display(states[state])
mean_df= pd.DataFrame(means.values(),index=means.keys(),columns=['mean_sum'])

state= mean_df.sort_values(by='mean_sum',ascending=False).iloc[0].name

print(f'"{state}" State has the highest dropout ratio')

mean_df
mean_2012_13={}

mean_2013_14={}

mean_2014_15={}



for state in states.keys():

    

    mean_2012_13[str(state)]= states[state].iloc[0]['mean']

    try:

        mean_2013_14[str(state)]= states[state].iloc[1]['mean']

    except IndexError as e:

        mean_2013_14[str(state)]= np.nan        

    try:

        mean_2014_15[str(state)]= states[state].iloc[2]['mean']

    except IndexError as e:

        mean_2014_15[str(state)]= np.nan

        

#     mean_2012_13[str(state)]= states[state].iloc[0]['mean']

#display(mean_2013_14[str(state)]= states[state].iloc[1]['mean'])

#     mean_2014_15[str(state)]= states[state].iloc[2]['mean']
mean_df['mean_2012_13']= mean_2012_13.values()

mean_df['mean_2013_14']= mean_2013_14.values()

mean_df['mean_2014_15']= mean_2014_15.values()

mean_df
cols= mean_df.isna().sum(axis=1)[mean_df.isna().sum(axis=1)>0].index

for i in range(len(cols)):

    mean_df.loc[cols[i]].fillna(method='ffill',inplace=True)



mean_df

mean_df[['mean_2012_13', 'mean_2013_14', 'mean_2014_15']].plot(kind='bar',figsize=(20,5))
data_water= pd.read_csv('../input/indian-school-education-statistics/percentage-of-schools-with-water-facility-2013-2016.csv',parse_dates=['Year'],index_col='Year')

data_water.columns= ['state',

                     'school_2_4',

                     'school_2_8',

                     'school_2_12',

                     'school_4_8',

                     'school_4_12',

                     'school_2_10',

                     'school_4_10',

                     'school_9_10',

                     'school_9_12',

                     'school_11_12',

                     'all_schools'

                    ]
state= 'Uttarakhand'

u= data_water.groupby('state').get_group(state)



z=(u[[ 'school_2_4', 'school_2_8', 'school_2_12', 'school_4_8',

       'school_4_12', 'school_2_10', 'school_4_10', 'school_9_10',

       'school_9_12', 'school_11_12']]-100)* (-1)



z.plot(kind='bar',figsize=(20,4))

plt.title(f'School levels in {state} which had poor water facility in 2013-2014-2015')
state= 'Manipur'

u= data_water.groupby('state').get_group(state)



z=(u[[ 'school_2_4', 'school_2_8', 'school_2_12', 'school_4_8',

       'school_4_12', 'school_2_10', 'school_4_10', 'school_9_10',

       'school_9_12', 'school_11_12']]-100)* (-1)



z.plot(kind='bar',figsize=(20,4))

plt.title(f'School levels in {state} which had poor water facility in 2013-2014-2015')
for state in data_water['state'].unique():

    u= data_water.groupby('state').get_group(state)



    z=(u[[ 'school_2_4', 'school_2_8', 'school_2_12', 'school_4_8',

           'school_4_12', 'school_2_10', 'school_4_10', 'school_9_10',

           'school_9_12', 'school_11_12']]-100)* (-1)



    z.plot(kind='bar',figsize=(20,4))

    plt.title(f'School levels in {state} which had poor water facility in 2013-2014-2015')

    