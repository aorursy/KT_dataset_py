# Setup

import os

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

init_notebook_mode(connected=True)

cf.go_offline()

%matplotlib inline
# Data file

path = '../input/market-price-of-onion-2020/Onion Prices 2020.csv'



# Reading data

Data = pd.read_csv(path)

Data.head()
# Extracting Date & Month from arrival_date

Data['Month'] = Data['arrival_date'].apply(lambda x :x.split('/')[1])

Data['Date'] = Data['arrival_date'].apply(lambda x :x.split('/')[0])
Data.head()
# So let's determine the shape of our data.

# Will also check for null values



print('Data Info :\n\tRow Count - {r:}\n\tColumn Count - {c:}\n\tNull Values - {n:}'.format(r=Data.shape[0],

                                            c=Data.shape[1], n=Data.isna().sum().sum()))



# Lets check for number of unique values for all categorical columns.

print('\nValue Count :')

for category in ['state', 'district', 'market', 'commodity', 'variety']:

    print("\t'{c:}s' mentioned : {n:}".format(c=category.title(), n=Data[category].nunique()))
# Min, Max & Modal Price Distribution

plt.figure(figsize=(12, 5))

sns.set_style('darkgrid')

sns.distplot(a=Data['min_price'], bins=80, color='green', hist=False, label='Minimum Price')

sns.distplot(a=Data['max_price'], bins=80, color='red', hist=False, label='Maximum Price')

sns.distplot(a=Data['modal_price'], bins=80, color='orange', hist=False, label='Modal Price')

plt.title('Various Price Distributions')

plt.xlabel('Price')

sns.despine()
# As we saw that onion prices were zero. let's explore more into it

print('Zero ₹ Onion Prices :\n\tState Count : {c:}\n\tState List : {s:}\n\tVariety List : {v:}\n'.format(c=Data[Data['min_price']==0]['state'].nunique(),

                                                                            s=list(Data[Data['min_price']==0]['state'].unique()),

                                                                            v=list(Data[Data['min_price']==0]['variety'].unique())))





# Also consider onion price to be 18000₹/Quintal. 

print('18000₹/Q Onion Price :\n\tState Count : {c:}\n\tState List : {s:}\n\tVariety List : {v:}'.format(c=Data[Data['min_price']==18000]['state'].nunique(),

                                                                            s=list(Data[Data['min_price']==18000]['state'].unique()),

                                                                            v=list(Data[Data['min_price']==18000]['variety'].unique())))
# Bar Plot

# X-axis : Months

# Y-axis : Price



fig, axes = plt.subplots(nrows=11, ncols=2, figsize=(20, 60), sharey=True)

fig.suptitle("Average Onion Price in States for each month of Year 2020", fontsize=24)

sns.set_style("darkgrid")



states_list = Data['state'].unique()     # List of all States

rows = [x for x in range(0, 11)]

cols = [0]

count = 1



for state in states_list:

    # bar plot

    state_fig = Data.groupby(['state', 'Month']).mean().xs(state).plot(kind='bar',ax=axes[rows[0], cols[0]])

    state_fig.set_title(state, fontdict={'fontsize': 20, 'color' : 'red'})

    state_fig.set_xticklabels(labels=state_fig.get_xticklabels(), rotation=360)

    

    # column switch

    if cols[0] == 0:

        cols[0] = 1

    else :

        cols[0] = 0

        

    # rows switch

    count += 1

    if count > 2:

        rows.pop(0)

        count = 1

    

    fig.tight_layout()

    fig.subplots_adjust(top=0.96)
# Line Plot

# X-axis : Arrival Dates

# Y-axis : Price



fig, axes = plt.subplots(nrows=22, ncols=1, figsize=(20, 100))

fig.suptitle("Average Onion Price in States on Arrival in Year 2020", fontsize=24)

sns.set_style("darkgrid")



states_list = Data['state'].unique()     # List of all States

rows = [x for x in range(0, 22)]

count = 1

                                                

for state in states_list:

    # bar plot

    state_fig = Data.sort_values(by=['Month', 'Date']).groupby(['state','arrival_date'], sort=False).mean().xs(state).reset_index().plot(kind='line',

                                                                        ax=axes[rows[0]], x='arrival_date', marker='o', markersize=3, markerfacecolor='black')

    state_fig.set_title(state, fontdict={'fontsize': 20, 'color' : 'red'})

    state_fig.set_xticklabels(labels=state_fig.get_xticklabels(), rotation=360)

    

    # rows switch

    rows.pop(0)

    

    fig.tight_layout()

    fig.subplots_adjust(top=0.96)
# States with Minimum Prices less than or equal to 500₹/Quintal

states = Data[Data['min_price'] <= 500]['state'].unique()

print('States with min. price <= 500₹')

for state in states:

    print('\t{s:}.'.format(s=state.title()))

#Data[(Data['min_price'] <= 500) & (Data['state']=='Andhra Pradesh')]['Month'].unique()
# Box Plot : Variety of Onions wrt Prices

# Count Plot : Variety of Onions wrt State





fig, axes = plt.subplots(nrows=21, ncols=2, figsize=(20, 100))

fig.suptitle("Variety of Onions & relation wrt Prices & State", fontsize=24)

sns.set_style("darkgrid")



variety_list = Data['variety'].unique()   # List of all onion variety

rows = [x for x in range(0, 21)]

cols = [0]

                                                

for variety in variety_list: 

    # box plot

    box_plot = sns.boxplot(data=Data[Data['variety']==variety][['min_price', 'max_price', 'modal_price']], ax=axes[rows[0], cols[0]])

    box_plot.set_title(variety, fontdict={'fontsize': 20, 'color' : 'red'})

    box_plot.set_xticklabels(labels=box_plot.get_xticklabels(), rotation=360)

    

    # cols switch for count plot

    cols[0] = 1

    

    # count plot

    count_plot = sns.countplot(Data[Data['variety']==variety]['state'], ax=axes[rows[0], cols[0]])

    count_plot.set_title(variety, fontdict={'fontsize': 20, 'color' : 'red'})

    count_plot.set_xticklabels(labels=count_plot.get_xticklabels(), rotation=25, fontdict={'fontsize': 8})

    

    # rows & cols switch for box plot

    rows.pop(0)

    cols[0] = 0

    

    fig.tight_layout()

    fig.subplots_adjust(top=0.96)
# Percentage change in prices between arrival days



fig, axes = plt.subplots(nrows=22, ncols=1, figsize=(20, 100))

fig.suptitle("Percentage Change in Average Onion Price in States considering Arrivals in Year 2020", fontsize=24)

sns.set_style("darkgrid")



states_list = Data['state'].unique()     # List of all States

rows = [x for x in range(0, 22)]

count = 1

                                                

for state in states_list:

    # bar plot

    state_fig = Data.sort_values(by=['Month', 'Date']).groupby(['state','arrival_date'], sort=False).mean().xs(state).pct_change().reset_index().plot(kind='line',

                                                                        ax=axes[rows[0]], x='arrival_date', marker='o', markersize=3, markerfacecolor='black')

    state_fig.set_title(state, fontdict={'fontsize': 20, 'color' : 'red'})

    state_fig.set_xticklabels(labels=state_fig.get_xticklabels(), rotation=360)

    

    # rows switch

    rows.pop(0)

    

    fig.tight_layout()

    fig.subplots_adjust(top=0.96)


