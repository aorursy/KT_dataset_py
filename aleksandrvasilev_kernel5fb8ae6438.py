# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
open_interest = pd.read_csv('../input/open-interest-10-08/derivatives-open-positions-list-2020807.csv')

open_interest_ind = pd.read_csv('../input/open-interest-10-08/derivatives-open-positions-list-2020807.csv', index_col="isin", parse_dates=True)

open_fil=open_interest_ind.fillna(0)

sort=open_fil.loc[open_fil.loc[:,'iz_fiz']

== 1].mean()

sort
open_interest_fill=open_interest.fillna(0)

open_loc_isna = open_interest_fill.loc[:, ['contract_type','name','isin', 'clients_in_long','long_position','clients_in_short', 'short_position']]

open_loc_isna

def remean_points(row):

    row.long_position = row.long_position - row.clients_in_long

    row.short_position = row.short_position - row.clients_in_short

    return row



open_loc = open_loc_isna.apply(remean_points, axis='columns')

open_loc


def filter_pos(data):

    index=[]

    for index in data:

        index= data.clients_in_long/data.clients_in_short

        

    return index

open_loc_app=filter_pos(open_loc).sort_values(ascending=False).head(10)

open_loc_app

print('Коэффициент покупки/продажи')

print(open_loc_app)

open_loc_sum=open_loc.groupby('isin').sum()

open_loc_sum_sort=open_loc_sum.sort_values(by='long_position')

Open_long_and_short_top10=open_loc_sum_sort.iloc[-10:]

Open_short_top10=Open_long_and_short_top10.loc[:,['clients_in_short','short_position']]

Open_long_top10=Open_long_and_short_top10.loc[:, ['clients_in_long','long_position']]

print('MAX')

print('clients_in_long=', open_loc_sum.clients_in_long.max())

print('clients_in_short=', open_loc_sum.clients_in_short.max())

print('long_position=', open_loc_sum.long_position.max())

print('short_position=', open_loc_sum.short_position.max())



open_loc_isna_min=open_loc_isna.groupby('isin').sum().median()

print('median')

print(open_loc_isna_min)

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

plt.figure(figsize=(14,6))

sns.kdeplot(data=Open_long_top10['clients_in_long'], label="clients_in_long", shade=True)

sns.kdeplot(data=Open_short_top10['clients_in_short'], label="clients_in_short", shade=True)

#sns.kdeplot(data=a['short_position'], label="clients_in_long", shade=True)

#sns.kdeplot(data=a['clients_in_short'], label="clients_in_long", shade=True)

# Add title

plt.title("Open Interest CLIENTS 07.10.2020")
plt.figure(figsize=(14,6))

sns.kdeplot(data=Open_long_top10['long_position'], label="long_position", shade=True)

sns.kdeplot(data=Open_short_top10['short_position'], label="short_position", shade=True)

#sns.kdeplot(data=Open_long_top10['clients_in_long'], label="clients_in_long", shade=True)

#sns.kdeplot(data=Open_short_top10['clients_in_short'], label="clients_in_short", shade=True)



# Add title

plt.title("Open Interest URIST 07.10.2020")


sns.set_style('darkgrid')

plt.figure(figsize=(14,6))

sns.lineplot(data=Open_long_top10['clients_in_long'], label="clients_in_long")

sns.lineplot(data=Open_short_top10['clients_in_short'], label="clients_in_short")

sns.set_style('darkgrid')

plt.figure(figsize=(14,6))

sns.lineplot(data=Open_long_top10['long_position'], label="long_position")

sns.lineplot(data=Open_short_top10['short_position'], label="short_position")
plt.figure(figsize=(18,8))



plt.title("Average Arrival")



sns.barplot(x=Open_long_top10.index, y=Open_long_top10['clients_in_long'])

sns.barplot(x=Open_short_top10.index, y=Open_short_top10['clients_in_short'])



plt.ylabel("clients_in_long/clients_in_short")

plt.figure(figsize=(18,8))



plt.title("Total_in_long")



#sns.barplot(x=Open_long_top10.index, y=Open_long_top10['clients_in_long'])

#sns.barplot(x=Open_short_top10.index, y=Open_short_top10['clients_in_short'])

sns.barplot(x=Open_long_top10.index, y=Open_long_top10['long_position'])

#sns.barplot(x=Open_short_top10.index, y=Open_short_top10['short_position'])



plt.ylabel("Total long")
plt.figure(figsize=(18,8))



plt.title("Total short")



#sns.barplot(x=Open_long_top10.index, y=Open_long_top10['clients_in_long'])

#sns.barplot(x=Open_short_top10.index, y=Open_short_top10['clients_in_short'])

#sns.barplot(x=Open_long_top10.index, y=Open_long_top10['long_position'])

sns.barplot(x=Open_short_top10.index, y=Open_short_top10['short_position'])



plt.ylabel("Total short")