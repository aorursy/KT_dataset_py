# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Getting the required packages for our analysis



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import datetime

import seaborn as sns # seaborn package for visualising

import plotly.express as px # plotly visualisation

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import plotly.graph_objects as go



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Reading the csv files from input files-

sell_prices=pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')

calendar=pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

sales_train_validation=pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')
# Checking initial data for their behaviour

sell_prices.head()
#pd.set_option('max_rows',None)

calendar
sales_train_validation.head()
plot=sns.countplot(data=sell_prices,x=sell_prices['store_id'])

# These are just value counts and data that is there contains time-series data so IT is possible,store_id would be repeating again

# and thus we need to investigate further.
#sell_prices[(sell_prices==max(sell_prices['wm_yr_wk']))]

print('The dataset contains {0} rows and {1} columns'.format(sell_prices.shape[0],sell_prices.shape[1]))
#Converting column names from object to category

sell_prices['store_id']=sell_prices['store_id'].astype('category')

sell_prices['item_id']=sell_prices['item_id'].astype('category')

sell_prices.info()

#Checking the combination of min(wm_yr_wk) and max(wm_yr_wk)

print('Data is currently available for minimum wm_yr_wk-{0} and maximum wm_yr_wk-{1}'.format(min(sell_prices['wm_yr_wk']),max(sell_prices['wm_yr_wk'])))

# 11101 --> first week ID

# 11621 --> last week ID 
#Converting to specified data

calendar['date']=pd.to_datetime(calendar['date'])

calendar['weekday']=calendar['weekday'].astype('category')

calendar['wday']=calendar['wday'].astype('category')

calendar['month']=calendar['month'].astype('category')

calendar['year']=calendar['year'].astype('category')
calendar.info()
print('Minimum of the date collected-{0} and the maximum of the date collected-{1}'.format(min(calendar['date']),max(calendar['date'])))
no_of_days=max(calendar['date'])-min(calendar['date'])

print('Total no. of weeks for data has been collected -',no_of_days.days//7)
mapping_id_date=calendar[['date','wm_yr_wk','weekday','wday','month','year']]

sell_prices_date=sell_prices.join(mapping_id_date.set_index('wm_yr_wk'),on='wm_yr_wk')
sell_prices_date.head()
#Dividing based on state

CA_data=sell_prices_date[sell_prices_date['store_id'].str.match('CA')]

TX_data=sell_prices_date[sell_prices_date['store_id'].str.match('TX')]

WI_data=sell_prices_date[sell_prices_date['store_id'].str.match('WI')]
CA_data.tail()
#Joining the id with the corresponding dates

CA_data_group=pd.DataFrame(CA_data.groupby(by=['year','store_id','item_id']).mean())

TX_data_group=pd.DataFrame(TX_data.groupby(by=['year','store_id','item_id']).mean())

WI_data_group=pd.DataFrame(WI_data.groupby(by=['year','store_id','item_id']).mean())
CA_data_group.tail()
CA_data_group=CA_data_group.reset_index()

TX_data_group=TX_data_group.reset_index()

WI_data_group=WI_data_group.reset_index()
CA_data_group=CA_data_group[['year', 'store_id', 'item_id','sell_price']]

TX_data_group=TX_data_group[['year', 'store_id', 'item_id','sell_price']]

WI_data_group=WI_data_group[['year', 'store_id', 'item_id','sell_price']]
#3 item groups - HOUSEHOLD,FOODS,HOBBIES

CA_data_group_HOUSEHOLD=CA_data_group[(CA_data_group['item_id'].str.match('HOUSEHOLD'))&(CA_data_group['store_id'].str.match('CA'))]

CA_data_group_FOODS=CA_data_group[(CA_data_group['item_id'].str.match('FOODS'))&(CA_data_group['store_id'].str.match('CA'))]

CA_data_group_HOBBIES=CA_data_group[(CA_data_group['item_id'].str.match('HOBBIES'))&(CA_data_group['store_id'].str.match('CA'))]



TX_data_group_HOUSEHOLD=TX_data_group[(TX_data_group['item_id'].str.match('HOUSEHOLD'))&(TX_data_group['store_id'].str.match('TX'))]

TX_data_group_FOODS=TX_data_group[(TX_data_group['item_id'].str.match('FOODS'))&(TX_data_group['store_id'].str.match('TX'))]

TX_data_group_HOBBIES=TX_data_group[(TX_data_group['item_id'].str.match('HOBBIES'))&(TX_data_group['store_id'].str.match('TX'))]



WI_data_group_HOUSEHOLD=WI_data_group[(WI_data_group['item_id'].str.match('HOUSEHOLD'))&(WI_data_group['store_id'].str.match('WI'))]

WI_data_group_FOODS=WI_data_group[(WI_data_group['item_id'].str.match('FOODS'))&(WI_data_group['store_id'].str.match('WI'))]

WI_data_group_HOBBIES=WI_data_group[(WI_data_group['item_id'].str.match('HOBBIES'))&(WI_data_group['store_id'].str.match('WI'))]
sell_price_avg = pd.concat([CA_data_group_HOUSEHOLD.rename(columns={'sell_price':'sell_price_CA_Household'}).groupby(['year']).mean(),

                  CA_data_group_FOODS.rename(columns={'sell_price':'sell_price_CA_Foods'}).groupby(['year']).mean(),

                 CA_data_group_HOBBIES.rename(columns={'sell_price':'sell_price_CA_Hobbies'}).groupby(['year']).mean(),

                 TX_data_group_HOUSEHOLD.rename(columns={'sell_price':'sell_price_TX_Household'}).groupby(['year']).mean(),

                 TX_data_group_FOODS.rename(columns={'sell_price':'sell_price_TX_Foods'}).groupby(['year']).mean(),

                 TX_data_group_HOBBIES.rename(columns={'sell_price':'sell_price_TX_Hobbies'}).groupby(['year']).mean(),

                 WI_data_group_HOUSEHOLD.rename(columns={'sell_price':'sell_price_WI_Household'}).groupby(['year']).mean(),

                 WI_data_group_FOODS.rename(columns={'sell_price':'sell_price_WI_Foods'}).groupby(['year']).mean(),

                 WI_data_group_HOBBIES.rename(columns={'sell_price':'sell_price_WI_Hobbies'}).groupby(['year']).mean()], axis=1)

sell_price_avg=sell_price_avg.reset_index()
sell_price_avg
sell_price_avg.columns
sns.lineplot(x='year',y='sell_price_CA_Household',data=sell_price_avg)

sns.lineplot(x='year',y='sell_price_CA_Foods',data=sell_price_avg)

sns.lineplot(x='year',y='sell_price_CA_Hobbies',data=sell_price_avg)

sns.lineplot(x='year',y='sell_price_TX_Household',data=sell_price_avg)

sns.lineplot(x='year',y='sell_price_TX_Foods',data=sell_price_avg)

sns.lineplot(x='year',y='sell_price_TX_Hobbies',data=sell_price_avg)

sns.lineplot(x='year',y='sell_price_WI_Household',data=sell_price_avg)

sns.lineplot(x='year',y='sell_price_WI_Foods',data=sell_price_avg)

sns.lineplot(x='year',y='sell_price_WI_Hobbies',data=sell_price_avg)





#sell_price_avg.info()
fig = go.Figure()

fig.add_trace(go.Scatter(x=sell_price_avg['year'], y=sell_price_avg['sell_price_CA_Household'],name='sell_price_CA_Household',line_shape='linear'))

fig.add_trace(go.Scatter(x=sell_price_avg['year'], y=sell_price_avg['sell_price_CA_Foods'], name="sell_price_CA_Foods",line_shape='linear'))

fig.add_trace(go.Scatter(x=sell_price_avg['year'], y=sell_price_avg['sell_price_CA_Hobbies'],name='sell_price_CA_Hobbies',line_shape='linear'))

fig.add_trace(go.Scatter(x=sell_price_avg['year'], y=sell_price_avg['sell_price_TX_Household'],name='sell_price_TX_Household',line_shape='linear'))

fig.add_trace(go.Scatter(x=sell_price_avg['year'], y=sell_price_avg['sell_price_TX_Foods'],name='sell_price_TX_Foods',line_shape='linear'))

fig.add_trace(go.Scatter(x=sell_price_avg['year'], y=sell_price_avg['sell_price_TX_Hobbies'],name='sell_price_TX_Hobbies',line_shape='linear'))

fig.add_trace(go.Scatter(x=sell_price_avg['year'], y=sell_price_avg['sell_price_WI_Household'],name='sell_price_WI_Household',line_shape='linear'))

fig.add_trace(go.Scatter(x=sell_price_avg['year'], y=sell_price_avg['sell_price_WI_Foods'],name='sell_price_WI_Foods',line_shape='linear'))

fig.add_trace(go.Scatter(x=sell_price_avg['year'], y=sell_price_avg['sell_price_WI_Hobbies'],name='sell_price_WI_Hobbies',line_shape='linear'))



fig.update_traces(hoverinfo='text+name', mode='lines+markers')

fig.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=16))



fig.show()
CA_data_group_HOUSEHOLD.info()
CA_data_group_HOUSEHOLD['item_id'].unique()
# CA_data_group_HOUSEHOLD['item_id'].unique()

# CA_data_group_FOODS['item_id'].unique() 

# CA_data_group_HOBBIES['item_id'].unique() 

# TX_data_group_HOUSEHOLD['item_id'].unique() 

# TX_data_group_FOODS['item_id'].unique() 

# TX_data_group_HOBBIES['item_id'].unique() 

# WI_data_group_HOUSEHOLD['item_id'].unique() 

# WI_data_group_FOODS['item_id'].unique() 

# WI_data_group_HOBBIES['item_id'].unique() 
#Analysis at store_cd level

# def store_level_graphs(store_id,data):

CA_data_group_HOUSEHOLD.head()
sell_price_avg_store_id_CA = pd.concat([CA_data_group_HOUSEHOLD.rename(columns={'sell_price':'sell_price_CA_Household'}).groupby(['year','store_id']).mean(),

                  CA_data_group_FOODS.rename(columns={'sell_price':'sell_price_CA_Foods'}).groupby(['year','store_id']).mean(),

                 CA_data_group_HOBBIES.rename(columns={'sell_price':'sell_price_CA_Hobbies'}).groupby(['year','store_id']).mean()], axis=1)

                 

sell_price_avg_store_id_TX=pd.concat([TX_data_group_HOUSEHOLD.rename(columns={'sell_price':'sell_price_TX_Household'}).groupby(['year','store_id']).mean(),

                 TX_data_group_FOODS.rename(columns={'sell_price':'sell_price_TX_Foods'}).groupby(['year','store_id']).mean(),

                 TX_data_group_HOBBIES.rename(columns={'sell_price':'sell_price_TX_Hobbies'}).groupby(['year','store_id']).mean()], axis=1)

sell_price_avg_store_id_WI=pd.concat([

                 WI_data_group_HOUSEHOLD.rename(columns={'sell_price':'sell_price_WI_Household'}).groupby(['year','store_id']).mean(),

                 WI_data_group_FOODS.rename(columns={'sell_price':'sell_price_WI_Foods'}).groupby(['year','store_id']).mean(),

                 WI_data_group_HOBBIES.rename(columns={'sell_price':'sell_price_WI_Hobbies'}).groupby(['year','store_id']).mean()], axis=1)
sell_price_avg_store_id_CA=sell_price_avg_store_id_CA.dropna().reset_index()

sell_price_avg_store_id_TX=sell_price_avg_store_id_TX.dropna().reset_index()

sell_price_avg_store_id_WI=sell_price_avg_store_id_WI.dropna().reset_index()
fig=px.line(sell_price_avg_store_id_CA, x="year", y="sell_price_CA_Household", color='store_id',title='Average of sell price of Household products across California')

fig.show()

fig=px.line(sell_price_avg_store_id_CA, x="year", y="sell_price_CA_Foods", color='store_id',title='Average of sell price of Food products across California')

fig.show()

fig=px.line(sell_price_avg_store_id_CA, x="year", y="sell_price_CA_Hobbies", color='store_id',title='Average of sell price of Hobbies across California')

fig.show()
fig=px.line(sell_price_avg_store_id_TX, x="year", y="sell_price_TX_Household", color='store_id',title='Average of sell price of Household products across Texas')

fig.show()

fig=px.line(sell_price_avg_store_id_TX, x="year", y="sell_price_TX_Foods", color='store_id',title='Average of sell price of Food products across Texas')

fig.show()

fig=px.line(sell_price_avg_store_id_TX, x="year", y="sell_price_TX_Hobbies", color='store_id',title='Average of sell price of Hobbies across Texas')

fig.show()


fig=px.line(sell_price_avg_store_id_WI, x="year", y="sell_price_WI_Household", color='store_id',title='Average of sell price of Household products across Wisconsin')

fig.show()

fig=px.line(sell_price_avg_store_id_WI, x="year", y="sell_price_WI_Foods", color='store_id',title='Average of sell price of Food products across Wisconsin')

fig.show()

fig=px.line(sell_price_avg_store_id_WI, x="year", y="sell_price_WI_Hobbies", color='store_id',title='Average of sell price of Hobbies across Wisconsin')

fig.show()
sell_price_avg_item_id_Household = pd.concat([CA_data_group_HOUSEHOLD.rename(columns={'sell_price':'sell_price_CA_Household'}).groupby(['year','item_id']).mean(),

                  TX_data_group_HOUSEHOLD.rename(columns={'sell_price':'sell_price_TX_Household'}).groupby(['year','item_id']).mean(),

                 WI_data_group_HOUSEHOLD.rename(columns={'sell_price':'sell_price_WI_Household'}).groupby(['year','item_id']).mean()], axis=1)



                 

sell_price_avg_item_id_Foods=pd.concat([CA_data_group_FOODS.rename(columns={'sell_price':'sell_price_CA_Foods'}).groupby(['year','item_id']).mean(),

                 TX_data_group_FOODS.rename(columns={'sell_price':'sell_price_TX_Foods'}).groupby(['year','item_id']).mean(),

                 WI_data_group_FOODS.rename(columns={'sell_price':'sell_price_WI_Foods'}).groupby(['year','item_id']).mean()], axis=1)

sell_price_avg_item_id_Hobbies=pd.concat([

                 CA_data_group_HOBBIES.rename(columns={'sell_price':'sell_price_CA_Hobbies'}).groupby(['year','item_id']).mean(),

                 TX_data_group_HOBBIES.rename(columns={'sell_price':'sell_price_TX_Hobbies'}).groupby(['year','item_id']).mean(),

                 WI_data_group_HOBBIES.rename(columns={'sell_price':'sell_price_WI_Hobbies'}).groupby(['year','item_id']).mean()], axis=1)
sell_price_avg_item_id_Household=sell_price_avg_item_id_Household.dropna().reset_index()

sell_price_avg_item_id_Foods=sell_price_avg_item_id_Foods.dropna().reset_index()

sell_price_avg_item_id_Hobbies=sell_price_avg_item_id_Hobbies.dropna().reset_index()
sell_price_avg_item_id_Household.columns
# Taking 3 examples of each of products which have high sell price and low sell price

# sell_price_avg_item_id_Household.sort_values(by=['sell_price_CA_Household','item_id'],ascending=False).head(15)

## Examples : HOUSEHOLD_1_060,HOUSEHOLD_2_446,HOUSEHOLD_1_378

#sell_price_avg_item_id_Household.sort_values(by=['sell_price_CA_Household','item_id']).head(15)

#Examples : HOUSEHOLD_2_371,HOUSEHOLD_1_151,HOUSEHOLD_1_517

#sell_price_avg_item_id_Household.sort_values(by=['sell_price_TX_Household','item_id'],ascending=False).head(15)

# Examples : HOUSEHOLD_1_060, HOUSEHOLD_2_446,HOUSEHOLD_1_378

#sell_price_avg_item_id_Household.sort_values(by=['sell_price_TX_Household','item_id']).head(15)

# Examples : HOUSEHOLD_2_371,HOUSEHOLD_1_151,HOUSEHOLD_1_503

# sell_price_avg_item_id_Household.sort_values(by=['sell_price_WI_Household','item_id'],ascending=False).head(15)

# Examples : HOUSEHOLD_1_060,HOUSEHOLD_2_446,HOUSEHOLD_1_378

#sell_price_avg_item_id_Household.sort_values(by=['sell_price_WI_Household','item_id']).head(15)

# Examples : HOUSEHOLD_2_371,HOUSEHOLD_1_151,HOUSEHOLD_1_517





# Taking 3 examples of each of products which have high sell price and low sell price

# sell_price_avg_item_id_Hobbies.sort_values(by=['sell_price_CA_Foods','item_id'],ascending=False).head(15)

#Examples: FOODS_3_298,FOODS_3_083,FOODS_2_239

# sell_price_avg_item_id_Foods.sort_values(by=['sell_price_CA_Foods','item_id']).head(15)

#Examples: FOODS_3_070,FOODS_3_580,FOODS_3_007

#sell_price_avg_item_id_Foods.sort_values(by=['sell_price_TX_Foods','item_id'],ascending=False).head(15)

#Examples: FOODS_3_298,FOODS_3_083,FOODS_2_389

# sell_price_avg_item_id_Foods.sort_values(by=['sell_price_TX_Foods','item_id']).head(15)

#Examples: FOODS_3_454,FOODS_3_007,FOODS_3_580\

# sell_price_avg_item_id_Foods.sort_values(by=['sell_price_WI_Foods','item_id'],ascending=False).head(15)

#Examples: FOODS_3_298,FOODS_3_083,FOODS_2_389

# sell_price_avg_item_id_Foods.sort_values(by=['sell_price_WI_Foods','item_id']).head(15)

#Examples: FOODS_3_547,FOODS_3_547,FOODS_3_547





# Taking 3 examples of each of products which have high sell price and low sell price

# sell_price_avg_item_id_Hobbies.sort_values(by=['sell_price_CA_Hobbies','item_id'],ascending=False).head(15)

#Examples: HOBBIES_1_361,HOBBIES_1_225,HOBBIES_1_060

# sell_price_avg_item_id_Hobbies.sort_values(by=['sell_price_CA_Hobbies','item_id']).head(15)

#Examples: HOBBIES_2_059,HOBBIES_2_142,HOBBIES_2_124

# sell_price_avg_item_id_Hobbies.sort_values(by=['sell_price_TX_Hobbies','item_id'],ascending=False).head(15)

#Examples: HOBBIES_1_410,HOBBIES_1_060,HOBBIES_1_361

#sell_price_avg_item_id_Hobbies.sort_values(by=['sell_price_TX_Hobbies','item_id']).head(15)

#Examples: HOBBIES_2_142,HOBBIES_2_129,HOBBIES_2_026

#sell_price_avg_item_id_Hobbies.sort_values(by=['sell_price_WI_Hobbies','item_id'],ascending=False).head(15)

#Examples: HOBBIES_1_361,HOBBIES_1_225,HOBBIES_1_060

#sell_price_avg_item_id_Hobbies.sort_values(by=['sell_price_WI_Hobbies','item_id']).head(15)

#Examples: HOBBIES_2_142,HOBBIES_2_129,HOBBIES_2_059

sell_price_avg_item_id_Household_eg=sell_price_avg_item_id_Household[(sell_price_avg_item_id_Household['item_id']=='HOUSEHOLD_1_060')|

                                (sell_price_avg_item_id_Household['item_id']=='HOUSEHOLD_2_446')|

                                (sell_price_avg_item_id_Household['item_id']=='HOUSEHOLD_1_378')|

                                (sell_price_avg_item_id_Household['item_id']=='HOUSEHOLD_2_371')|

                                (sell_price_avg_item_id_Household['item_id']=='HOUSEHOLD_1_151')|

                                (sell_price_avg_item_id_Household['item_id']=='HOUSEHOLD_1_517')|

                                (sell_price_avg_item_id_Household['item_id']=='HOUSEHOLD_1_503')]

sell_price_avg_item_id_Foods_eg=sell_price_avg_item_id_Foods[

    (sell_price_avg_item_id_Foods['item_id']=='FOODS_3_298')|

    (sell_price_avg_item_id_Foods['item_id']=='FOODS_3_083')|

    (sell_price_avg_item_id_Foods['item_id']=='FOODS_2_239')|

    (sell_price_avg_item_id_Foods['item_id']=='FOODS_3_070')|

    (sell_price_avg_item_id_Foods['item_id']=='FOODS_3_580')|

    (sell_price_avg_item_id_Foods['item_id']=='FOODS_3_007')|

    (sell_price_avg_item_id_Foods['item_id']=='FOODS_2_389')|

    (sell_price_avg_item_id_Foods['item_id']=='FOODS_3_454')|

    (sell_price_avg_item_id_Foods['item_id']=='FOODS_3_547')]



sell_price_avg_item_id_Hobbies_eg=sell_price_avg_item_id_Hobbies[

    (sell_price_avg_item_id_Hobbies['item_id']=='HOBBIES_1_361')|

     (sell_price_avg_item_id_Hobbies['item_id']=='HOBBIES_1_225')|

     (sell_price_avg_item_id_Hobbies['item_id']=='HOBBIES_1_060')|

     (sell_price_avg_item_id_Hobbies['item_id']=='HOBBIES_2_059')|

     (sell_price_avg_item_id_Hobbies['item_id']=='HOBBIES_2_142')|

     (sell_price_avg_item_id_Hobbies['item_id']=='HOBBIES_2_124')|

     (sell_price_avg_item_id_Hobbies['item_id']=='HOBBIES_1_410')|

     (sell_price_avg_item_id_Hobbies['item_id']=='HOBBIES_2_129')|

     (sell_price_avg_item_id_Hobbies['item_id']=='HOBBIES_2_026')

]

fig=px.line(sell_price_avg_item_id_Household_eg, x="year", y="sell_price_CA_Household", color='item_id',title='Average of sell price of Household products across California')

fig.show()

fig=px.line(sell_price_avg_item_id_Household_eg, x="year", y="sell_price_TX_Household", color='item_id',title='Average of sell price of Household products across Texas')

fig.show()

fig=px.line(sell_price_avg_item_id_Household_eg, x="year", y="sell_price_WI_Household", color='item_id',title='Average of sell price of Household products across Wisconsin')

fig.show()
fig=px.line(sell_price_avg_item_id_Foods_eg, x="year", y="sell_price_CA_Foods", color='item_id',title='Average of sell price of Food products across California')

fig.show()

fig=px.line(sell_price_avg_item_id_Foods_eg, x="year", y="sell_price_TX_Foods", color='item_id',title='Average of sell price of Food products across Texas')

fig.show()

fig=px.line(sell_price_avg_item_id_Foods_eg, x="year", y="sell_price_WI_Foods", color='item_id',title='Average of sell price of Food products across Wisconsin')

fig.show()
fig=px.line(sell_price_avg_item_id_Hobbies_eg, x="year", y="sell_price_CA_Hobbies", color='item_id',title='Average of sell price of Hobbies across California')

fig.show()

fig=px.line(sell_price_avg_item_id_Hobbies_eg, x="year", y="sell_price_TX_Hobbies", color='item_id',title='Average of sell price of Hobbies across Texas')

fig.show()

fig=px.line(sell_price_avg_item_id_Hobbies_eg, x="year", y="sell_price_WI_Hobbies", color='item_id',title='Average of sell price of Hobbies across Wisconsin')

fig.show()
#Not on basis of region but dividing based on year and then on first on monthly basis

print('Minimum of year {0} and maximum of year {1}'.format(min(sell_prices_date['year']),max(sell_prices_date['year'])))

sell_prices_date_2011=sell_prices_date[sell_prices_date['year']==2011]

sell_prices_date_2012=sell_prices_date[sell_prices_date['year']==2012]

sell_prices_date_2013=sell_prices_date[sell_prices_date['year']==2013]

sell_prices_date_2014=sell_prices_date[sell_prices_date['year']==2014]

sell_prices_date_2015=sell_prices_date[sell_prices_date['year']==2015]

sell_prices_date_2016=sell_prices_date[sell_prices_date['year']==2016]

sell_prices_date_2017=sell_prices_date[sell_prices_date['year']==2017]
sell_prices_date_2011_month=sell_prices_date_2011.groupby(by=['month','item_id']).mean().reset_index()

sell_prices_date_2012_month=sell_prices_date_2012.groupby(by=['month','item_id']).mean().reset_index()

sell_prices_date_2013_month=sell_prices_date_2013.groupby(by=['month','item_id']).mean().reset_index()

sell_prices_date_2014_month=sell_prices_date_2014.groupby(by=['month','item_id']).mean().reset_index()

sell_prices_date_2015_month=sell_prices_date_2015.groupby(by=['month','item_id']).mean().reset_index()

sell_prices_date_2016_month=sell_prices_date_2016.groupby(by=['month','item_id']).mean().reset_index()
sell_prices_date_2011_month=sell_prices_date_2011_month[['month','item_id','sell_price']]

sell_prices_date_2012_month=sell_prices_date_2012_month[['month','item_id','sell_price']]

sell_prices_date_2013_month=sell_prices_date_2013_month[['month','item_id','sell_price']]

sell_prices_date_2014_month=sell_prices_date_2014_month[['month','item_id','sell_price']]

sell_prices_date_2015_month=sell_prices_date_2015_month[['month','item_id','sell_price']]

sell_prices_date_2016_month=sell_prices_date_2016_month[['month','item_id','sell_price']]
def build_graph(data_set,title=None):

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data_set['month'], y=data_set['sell_price_Household'],name='sell_price_Household',line_shape='linear'))

    fig.add_trace(go.Scatter(x=data_set['month'], y=data_set['sell_price_Foods'], name="sell_price_Foods",line_shape='linear'))

    fig.add_trace(go.Scatter(x=data_set['month'], y=data_set['sell_price_Hobbies'],name='sell_price_Hobbies',line_shape='linear'))

    fig.update_traces(hoverinfo='text+name', mode='lines+markers')

    fig.update_layout(title=title,xaxis_title='Month',yaxis_title='Sell price',legend=dict(y=0.5, traceorder='reversed', font_size=16))

    

    fig.show()
sell_prices_date_2011_month_Household=sell_prices_date_2011_month[sell_prices_date_2011_month['item_id'].str.match('HOUSEHOLD')]

sell_prices_date_2011_month_Foods=sell_prices_date_2011_month[sell_prices_date_2011_month['item_id'].str.match('FOODS')]

sell_prices_date_2011_month_Hobbies=sell_prices_date_2011_month[sell_prices_date_2011_month['item_id'].str.match('HOBBIES')]
sell_prices_date_2011_month=pd.concat([sell_prices_date_2011_month_Household.rename(columns={'sell_price':'sell_price_Household'}).groupby(['month']).mean(),

                                      sell_prices_date_2011_month_Foods.rename(columns={'sell_price':'sell_price_Foods'}).groupby(['month']).mean(),

                                      sell_prices_date_2011_month_Hobbies.rename(columns={'sell_price':'sell_price_Hobbies'}).groupby(['month']).mean()],axis=1).reset_index()
build_graph(sell_prices_date_2011_month,'Sell prices for Year 2011')
sell_prices_date_2012_month_Household=sell_prices_date_2012_month[sell_prices_date_2012_month['item_id'].str.match('HOUSEHOLD')]

sell_prices_date_2012_month_Foods=sell_prices_date_2012_month[sell_prices_date_2012_month['item_id'].str.match('FOODS')]

sell_prices_date_2012_month_Hobbies=sell_prices_date_2012_month[sell_prices_date_2012_month['item_id'].str.match('HOBBIES')]
sell_prices_date_2012_month=pd.concat([sell_prices_date_2012_month_Household.rename(columns={'sell_price':'sell_price_Household'}).groupby(['month']).mean(),

                                      sell_prices_date_2012_month_Foods.rename(columns={'sell_price':'sell_price_Foods'}).groupby(['month']).mean(),

                                      sell_prices_date_2012_month_Hobbies.rename(columns={'sell_price':'sell_price_Hobbies'}).groupby(['month']).mean()],axis=1).reset_index()
build_graph(sell_prices_date_2012_month,'Sell prices for Year 2012')
sell_prices_date_2013_month_Household=sell_prices_date_2013_month[sell_prices_date_2013_month['item_id'].str.match('HOUSEHOLD')]

sell_prices_date_2013_month_Foods=sell_prices_date_2013_month[sell_prices_date_2013_month['item_id'].str.match('FOODS')]

sell_prices_date_2013_month_Hobbies=sell_prices_date_2013_month[sell_prices_date_2013_month['item_id'].str.match('HOBBIES')]

sell_prices_date_2013_month=pd.concat([sell_prices_date_2013_month_Household.rename(columns={'sell_price':'sell_price_Household'}).groupby(['month']).mean(),

                                      sell_prices_date_2013_month_Foods.rename(columns={'sell_price':'sell_price_Foods'}).groupby(['month']).mean(),

                                      sell_prices_date_2013_month_Hobbies.rename(columns={'sell_price':'sell_price_Hobbies'}).groupby(['month']).mean()],axis=1).reset_index()

build_graph(sell_prices_date_2013_month,'Sell prices for Year 2013')
sell_prices_date_2014_month_Household=sell_prices_date_2014_month[sell_prices_date_2014_month['item_id'].str.match('HOUSEHOLD')]

sell_prices_date_2014_month_Foods=sell_prices_date_2014_month[sell_prices_date_2014_month['item_id'].str.match('FOODS')]

sell_prices_date_2014_month_Hobbies=sell_prices_date_2014_month[sell_prices_date_2014_month['item_id'].str.match('HOBBIES')]
sell_prices_date_2014_month=pd.concat([sell_prices_date_2014_month_Household.rename(columns={'sell_price':'sell_price_Household'}).groupby(['month']).mean(),

                                      sell_prices_date_2014_month_Foods.rename(columns={'sell_price':'sell_price_Foods'}).groupby(['month']).mean(),

                                      sell_prices_date_2014_month_Hobbies.rename(columns={'sell_price':'sell_price_Hobbies'}).groupby(['month']).mean()],axis=1).reset_index()

build_graph(sell_prices_date_2014_month,'Sell prices for Year 2014')
sell_prices_date_2015_month_Household=sell_prices_date_2015_month[sell_prices_date_2015_month['item_id'].str.match('HOUSEHOLD')]

sell_prices_date_2015_month_Foods=sell_prices_date_2015_month[sell_prices_date_2015_month['item_id'].str.match('FOODS')]

sell_prices_date_2015_month_Hobbies=sell_prices_date_2015_month[sell_prices_date_2015_month['item_id'].str.match('HOBBIES')]



sell_prices_date_2015_month=pd.concat([sell_prices_date_2015_month_Household.rename(columns={'sell_price':'sell_price_Household'}).groupby(['month']).mean(),

                                      sell_prices_date_2015_month_Foods.rename(columns={'sell_price':'sell_price_Foods'}).groupby(['month']).mean(),

                                      sell_prices_date_2015_month_Hobbies.rename(columns={'sell_price':'sell_price_Hobbies'}).groupby(['month']).mean()],axis=1).reset_index()







build_graph(sell_prices_date_2015_month,'Sell prices for Year 2015')
sell_prices_date_2016_month_Household=sell_prices_date_2016_month[sell_prices_date_2016_month['item_id'].str.match('HOUSEHOLD')]

sell_prices_date_2016_month_Foods=sell_prices_date_2016_month[sell_prices_date_2016_month['item_id'].str.match('FOODS')]

sell_prices_date_2016_month_Hobbies=sell_prices_date_2016_month[sell_prices_date_2016_month['item_id'].str.match('HOBBIES')]



sell_prices_date_2016_month=pd.concat([sell_prices_date_2016_month_Household.rename(columns={'sell_price':'sell_price_Household'}).groupby(['month']).mean(),

                                      sell_prices_date_2016_month_Foods.rename(columns={'sell_price':'sell_price_Foods'}).groupby(['month']).mean(),

                                      sell_prices_date_2016_month_Hobbies.rename(columns={'sell_price':'sell_price_Hobbies'}).groupby(['month']).mean()],axis=1).reset_index()







build_graph(sell_prices_date_2016_month,'Sell prices for Year 2016')



submission=pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')

submission.shape
submission.tail()