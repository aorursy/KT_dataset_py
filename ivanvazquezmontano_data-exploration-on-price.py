# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

%matplotlib inline 

import seaborn as sns
print(os.listdir('../input'))



soccer_df = pd.read_csv('../input/footballsoccer-players-market-value/mostvaluableplayers.csv', delimiter=',',index_col='name')

soccer_df.dataframeName = 'mostvaluableplayers.csv'



soccer_df.head()
soccer_df.dtypes
soccer_df.shape
#Check for missing values

soccer_df.isnull().sum()
#Remove unnecessary column

soccer_df.drop(['#'],inplace=True,axis=1)



#Rename price column

soccer_df.rename(columns={'price_Mill._€': 'price'},inplace=True)



soccer_df.head()
#Summary of Price

soccer_df['price'].describe(percentiles=[.25,.5,.75])
#Distribution Plot of Price

sns.set(rc={'figure.figsize':(10,8)})

price_distn_plot = sns.distplot(soccer_df['price'], kde=False,bins = 20)

plt.title("Distribution Plot for Price")

price_distn_plot
#Price by position

soccer_df.groupby(['position'])['price'].count().sort_values(ascending=False)
#Create new column with general position for each player

soccer_df['general_pos'] = soccer_df.position



soccer_df['general_pos'].replace(to_replace=['Centre-Forward','Left Winger','Right Winger','Second Striker']\

                                 ,value='Forward',inplace=True)



soccer_df['general_pos'].replace(to_replace=['Central Midfield','Left Midfield','Right Midfield','Attacking Midfield','Defensive Midfield']\

                                 ,value='Midfield',inplace=True)



soccer_df['general_pos'].replace(to_replace=['Centre-Back','Left-Back','Right-Back','Second Striker']\

                                 ,value='Defense',inplace=True)



soccer_df.tail()
#Price by General Position



soccer_df.groupby(['general_pos'])['price'].count().sort_values(ascending=False)
#Avg by General Position

soccer_df.groupby(['general_pos'])['price'].mean().sort_values(ascending=False)
#Forward Price

forward_df = soccer_df.loc[soccer_df['general_pos'] == 'Forward']

forward_price = forward_df[['price']]



#Midfield Price

mid_df = soccer_df.loc[soccer_df['general_pos'] == 'Midfield']

mid_price = mid_df[['price']]



#Defense Price

defense_df = soccer_df.loc[soccer_df['general_pos'] == 'Defense']

defense_price = defense_df[['price']]



#Keeper Price

keeper_df = soccer_df.loc[soccer_df['general_pos'] == 'Goalkeeper']

keeper_price = keeper_df[['price']]



#All price in a List

price_by_pos = [forward_price,mid_price,defense_price,keeper_price]
#Empty list to append

p_b_p=[]



#List with values for column names

position = ['Forward','Midfield','Defense','Goalkeeper']



#for loop to get summary statistics for each position

for x in price_by_pos:

    i=x.describe(percentiles=[.25,.50,.75])

    i.reset_index(inplace=True)

    i.rename(columns={'index':'Stats'},inplace=True)

    p_b_p.append(i)

    
#rename columns

p_b_p[0].rename(columns={'price':position[0]},inplace=True)

p_b_p[1].rename(columns={'price':position[1]},inplace=True)

p_b_p[2].rename(columns={'price':position[2]},inplace=True)

p_b_p[3].rename(columns={'price':position[3]},inplace=True)



#View Table

position_summary = p_b_p

position_summary =[df.set_index('Stats') for df in position_summary]

position_summary = position_summary[0].join(position_summary[1:])

position_summary
#Violin plot to show distribution of prices for each position

sns.set(rc={'figure.figsize':(15,10)})

v_plot = sns.violinplot(x='general_pos',y='price',data=soccer_df,scale='area')

plt.title('Density & Distribution of price for each position')

v_plot
#Players getting paid atleast €100 

over_100 = soccer_df.loc[(soccer_df['price'] >= 100)]

over_100.groupby('general_pos').size()
over_100