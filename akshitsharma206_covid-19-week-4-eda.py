# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#importing the required libraries

import math

import numpy as np

import pandas as pd

from tqdm import tqdm

import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib.dates as dates

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

import numpy as np 

import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd

from datetime import datetime

from sklearn import preprocessing

from xgboost import XGBRegressor

from sklearn.tree import DecisionTreeRegressor

from lightgbm import LGBMRegressor

from catboost import CatBoostRegressor

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor 



# Suppress warnings

import warnings

warnings.filterwarnings("ignore")



PUBLIC_PRIVATE = 1 # 0 for public leaderboard, 1 for private



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train    = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

test     = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')
#Checking the first 5 rows

train.head()
#Checking the first 5 rows

test.head()
#Checking the affected cases

case = train.groupby('Country_Region')['ConfirmedCases'].max().sort_values(ascending=False).to_frame()

case.style.background_gradient(cmap='GnBu')
#Taking the top 10 countries

case=case.reset_index().head(10)
#Plotting a bar chart for top 10 countries

plt.figure(figsize =(10,10))

sns.set_color_codes("muted")

chart=sns.barplot(x='Country_Region',y='ConfirmedCases',data=case,label="Total Number of Cases")

for item in chart.get_xticklabels():

    item.set_rotation(90)
#making scatterplot using plotly

fig = go.Figure(data=go.Scatter(x=case['Country_Region'], y=case['ConfirmedCases'],

mode='lines+markers',hovertemplate = "Cases: %{y}<br> %{x}<extra></extra>",showlegend = False))

fig.show()
#Creating some new variables

train['MortalityRate'] = train['Fatalities'] / train['ConfirmedCases']

train['NewFatalities'] = train['Fatalities'] - train['Fatalities'].shift(1)

train['NewFatalities'] = train['NewFatalities'].fillna(0.0)

train['MortalityRate']= train['MortalityRate'].fillna(0.0)

train['NewConfirmedCases'] = train['ConfirmedCases'] - train['ConfirmedCases'].shift(1)

train['NewConfirmedCases'] = train['NewConfirmedCases'].fillna(0.0)

train['GrowthRate'] = train['NewConfirmedCases']/train['NewConfirmedCases'].shift(1)

train['GrowthRate']= train['GrowthRate'].replace([-np.inf, np.inf],  0.0)

train['GrowthRate']= train['GrowthRate'].fillna(0.0) 

train.head()
train_data_by_country = train.groupby(['Date','Country_Region'],as_index=False).agg({'ConfirmedCases': 'sum', 'Fatalities': 'sum',

                                                                                         'GrowthRate':'mean' })

max_train_date = train['Date'].max()

train_data_by_country_confirm = train_data_by_country.query('(Date == @max_train_date) & (ConfirmedCases > 100)').sort_values('ConfirmedCases', ascending=False)

train_data_by_country_confirm.set_index('Country_Region', inplace=True)

display(train_data_by_country_confirm.head())



from itertools import cycle, islice

discrete_col = list(islice(cycle(['orange', 'r', 'g', 'k', 'b', 'c', 'm']), None, len(train_data_by_country_confirm.head(30))))

plt.rcParams.update({'font.size': 22})

train_data_by_country_confirm.head(20).plot(figsize=(20,15), kind='barh', color=discrete_col)

plt.legend(["Confirmed Cases", "Fatalities"]);

plt.xlabel("Number of Covid-19 Affectees")

plt.title("First 20 Countries with Highest Confirmed Cases")

ylocs, ylabs = plt.yticks()

for i, v in enumerate(train_data_by_country_confirm.head(20)["ConfirmedCases"][:]):

    plt.text(v+0.01, ylocs[i]-0.25, str(int(v)), fontsize=12)

for i, v in enumerate(train_data_by_country_confirm.head(20)["Fatalities"][:]):

    if v > 0: #disply for only >300 fatalities

        plt.text(v+0.01,ylocs[i]+0.1,str(int(v)),fontsize=12)
def reformat_time(reformat, ax):

    ax.xaxis.set_major_locator(dates.WeekdayLocator())

    ax.xaxis.set_major_formatter(dates.DateFormatter('%b %d'))    

    if reformat: #reformat again if you wish

        date_list = train_data_by_date.reset_index()["Date"].tolist()

        x_ticks = [dt.datetime.strftime(t,'%Y-%m-%d') for t in date_list]

        x_ticks = [tick for i,tick in enumerate(x_ticks) if i%8==0 ]# split labels into same number of ticks as by pandas

        ax.set_xticklabels(x_ticks, rotation=90)

    # cosmetics

    ax.yaxis.grid(linestyle='dotted')

    ax.spines['right'].set_color('none')

    ax.spines['top'].set_color('none')

    ax.spines['left'].set_color('none')

    ax.spines['bottom'].set_color('none')



train['Date'] = pd.to_datetime(train['Date'])

train_data_by_date = train.groupby(['Date'],as_index=True).agg({'ConfirmedCases': 'sum',

                                                                'Fatalities': 'sum', 

                                                                'NewConfirmedCases':'sum', 

                                                                'NewFatalities':'sum', 

                                                                'MortalityRate':'mean'})

num0 = train_data_by_date._get_numeric_data() 

num0[num0 < 0.0] = 0.0

#display(train_data_by_date.head())



## ======= Sort by countries with fatalities > 500 ========



train_data_by_country_max = train.groupby(['Country_Region'],as_index=True).agg({'ConfirmedCases': 'max', 

                                                                                 'Fatalities': 'max'})

train_data_by_country_fatal = train_data_by_country_max[train_data_by_country_max['Fatalities']>500]

train_data_by_country_fatal = train_data_by_country_fatal.sort_values(by=['Fatalities'],ascending=False).reset_index()

display(train_data_by_country_fatal.head(20))



df_merge_by_country = pd.merge(train,train_data_by_country_fatal['Country_Region'],on=['Country_Region'],how='inner')

df_max_fatality_country = df_merge_by_country.groupby(['Date','Country_Region'],as_index=False).agg({'ConfirmedCases': 'sum',

                                                                                                     'Fatalities': 'sum',

                                                                                                     'NewConfirmedCases':'sum',

                                                                                                     'NewFatalities':'sum',

                                                                                                     'MortalityRate':'mean'})



num1 = df_max_fatality_country._get_numeric_data() 

num1[num1 < 0.0] = 0.0

df_max_fatality_country.set_index('Date',inplace=True)

#display(df_max_fatality_country.head(20))



countries = train_data_by_country_fatal['Country_Region'].unique()



plt.rcParams.update({'font.size': 16})



fig,(ax0,ax1) = plt.subplots(1,2,figsize=(15, 8))

fig,(ax2,ax3) = plt.subplots(1,2,figsize=(15, 8))#,sharey=True)



train_data_by_date.ConfirmedCases.plot(ax=ax0, x_compat=True, title='Confirmed Cases Globally', legend='Confirmed Cases',

                                       color=discrete_col)#, logy=True)

reformat_time(0,ax0)

train_data_by_date.NewConfirmedCases.plot(ax=ax0, x_compat=True, linestyle='dotted', legend='New Confirmed Cases',

                                          color=discrete_col)#, logy=True)

reformat_time(0,ax0)



train_data_by_date.Fatalities.plot(ax=ax2, x_compat=True, title='Fatalities Globally', legend='Fatalities', color='r')

reformat_time(0,ax2)

train_data_by_date.NewFatalities.plot(ax=ax2, x_compat=True, linestyle='dotted', legend='Daily Deaths',color='r')#tell pandas not to use its own datetime format

reformat_time(0,ax2)



for country in countries:

    match = df_max_fatality_country.Country_Region==country

    df_fatality_by_country = df_max_fatality_country[match] 

    df_fatality_by_country.ConfirmedCases.plot(ax=ax1, x_compat=True, title='Cumulative Confirmed Cases Nationally')

    reformat_time(0,ax1)

    df_fatality_by_country.Fatalities.plot(ax=ax3, x_compat=True, title='Cumulative Fatalities Nationally')

    reformat_time(0,ax3)

    

ax1.legend(countries)

ax3.legend(countries)
fig = plt.figure()

fig,(ax4,ax5) = plt.subplots(1,2,figsize=(15, 8))

#train_data_by_date.loc[(train_data_by_date.ConfirmedCases > 200)]#useless, its already summed.

train_data_by_date.MortalityRate.plot(ax=ax4, x_compat=True, legend='Mortality Rate',color='r')#tell pandas not to use its own datetime format

reformat_time(0,ax4)



for num, country in enumerate(countries):

    match = df_max_fatality_country.Country_Region==country 

    df_fatality_by_country = df_max_fatality_country[match] 

    df_fatality_by_country.MortalityRate.plot(ax=ax5, x_compat=True, title='Average Mortality Rate Nationally')    

    reformat_time(0,ax5)

ax5.legend(countries, loc='center left',bbox_to_anchor=(1.0, 0.5))
world_df = train_data_by_country.query('Date == @max_train_date')

world_df.loc[:,'Date']           = world_df.loc[:,'Date'].apply(str)

world_df.loc[:,'Confirmed_log']  = round(np.log10(world_df.loc[:,'ConfirmedCases'] + 1), 3)

world_df.loc[:,'Fatalities_log'] = np.log10(world_df.loc[:,'Fatalities'] + 1)

world_df.loc[:,'MortalityRate']  = round(world_df.loc[:, 'Fatalities'] / world_df.loc[:,'ConfirmedCases'], 3)

world_df.loc[:,'AveGrowthFactor']  = round(world_df.loc[:,'GrowthRate'], 3)

world_df.drop(['GrowthRate'], axis=1) #drop as this is actually the average over all the 74 days

display(world_df.head())



fig1 = px.choropleth(world_df, locations="Country_Region", 

                    locationmode="country names",  

                    color="Confirmed_log",                     

                    hover_name="Country_Region",

                    hover_data=['ConfirmedCases', 'Fatalities', 'MortalityRate', 'AveGrowthFactor'],

                    range_color=[world_df['Confirmed_log'].min(), world_df['Confirmed_log'].max()], 

                    color_continuous_scale = px.colors.sequential.Plasma,

                    title='COVID-19: Confirmed Cases')

fig1.show()



fig2 = px.scatter_geo(world_df, locations="Country_Region", 

                     locationmode="country names", 

                     color="ConfirmedCases", size='ConfirmedCases', 

                     hover_name="Country_Region", 

                     hover_data=['ConfirmedCases', 'Fatalities', 'MortalityRate', 'AveGrowthFactor'],

                     range_color= [world_df['Confirmed_log'].min(), world_df['ConfirmedCases'].max()], 

                     projection="natural earth", 

                     animation_frame="Date",

                     color_continuous_scale="portland",

                     title='COVID-19: Spread Over Time')

fig2.update(layout_coloraxis_showscale=False)

fig2.show()
# Taking care of the Null values

#Makinga a function for Null Values

EMPTY_VAL = "EMPTY_VAL"

def fillState(state, country):

    if state == EMPTY_VAL: return country

    return state
train['Province_State'].fillna(EMPTY_VAL, inplace=True)

train['Province_State'] = train.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)

test['Province_State'].fillna(EMPTY_VAL, inplace=True)

test['Province_State'] = test.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)
#Checking the null values

train.isnull().sum()
#Checking the null values

test.isnull().sum()