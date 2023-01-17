import gc

import os

from pathlib import Path

import random

import sys



from tqdm.notebook import tqdm

import numpy as np

import pandas as pd

import scipy as sp



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt

import seaborn as sns



from IPython.core.display import display, HTML



# --- plotly ---

from plotly import tools, subplots

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

import plotly.figure_factory as ff

import plotly.io as pio

pio.templates.default = "plotly_dark"



# --- models ---

from sklearn import preprocessing

from sklearn.model_selection import KFold

import lightgbm as lgb

import xgboost as xgb

import catboost as cb



# --- setup ---

pd.set_option('max_columns', 50)



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from itertools import cycle, islice

import seaborn as sb

import matplotlib.dates as dates

import datetime as dt



import plotly.offline as py

py.init_notebook_mode(connected=True)

#from plotly import tools, subplots

#import plotly.figure_factory as ff

import plotly.express as px

import plotly.graph_objects as go



train_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")#index_col=0

display(train_data.head())

test_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")#index_col=0

display(test_data.head())
sum_df = pd.pivot_table(train_data, values=['ConfirmedCases','Fatalities'], index=['Date'],aggfunc=np.sum)

display(sum_df.max())
train_data['NewConfirmedCases'] = train_data['ConfirmedCases'] - train_data['ConfirmedCases'].shift(1)

train_data['NewConfirmedCases'] = train_data['NewConfirmedCases'].fillna(0.0)

train_data['NewFatalities']     = train_data['Fatalities'] - train_data['Fatalities'].shift(1)

train_data['NewFatalities']     = train_data['NewFatalities'].fillna(0.0)#.astype(int)

train_data['MortalityRate']     = train_data['Fatalities'] / train_data['ConfirmedCases']

train_data['MortalityRate']     = train_data['MortalityRate'].fillna(0.0)

train_data['GrowthRate']        = train_data['NewConfirmedCases']/train_data['NewConfirmedCases'].shift(1)

train_data['GrowthRate']        = train_data['GrowthRate'].replace([-np.inf, np.inf],  0.0)

train_data['GrowthRate']        = train_data['GrowthRate'].fillna(0.0) 

display(train_data.head())
def getColumnInfo(df):

    n_province =  df['Province_State'].nunique()

    n_country  =  df['Country_Region'].nunique()

    n_days     =  df['Date'].nunique()

    start_date =  df['Date'].unique()[0]

    end_date   =  df['Date'].unique()[-1]

    return n_province, n_country, n_days, start_date, end_date



n_train = train_data.shape[0]

n_test = test_data.shape[0]



n_prov_train, n_count_train, n_train_days, start_date_train, end_date_train = getColumnInfo(train_data)

n_prov_test,  n_count_test,  n_test_days,  start_date_test,  end_date_test  = getColumnInfo(test_data)



print ('<==Train data==> \n # of Province_State: '+str(n_prov_train),', # of Country_Region:'+str(n_count_train), 

       ', Time Period: '+str(start_date_train)+' to '+str(end_date_train), '==> days:',str(n_train_days))

print("\n Countries with Province/State information:  ", train_data[train_data['Province_State'].isna()==False]['Country_Region'].unique())

print ('\n <==Test  data==> \n # of Province_State: '+str(n_prov_test),', # of Country_Region:'+str(n_count_test),

       ', Time Period: '+start_date_test+' to '+end_date_test, '==> days:',n_test_days)



df_test = test_data.loc[test_data.Date > '2020-04-03']

overlap_days = n_test_days - df_test.Date.nunique()

print('\n overlap days with training data: ', overlap_days, ', total days: ', n_train_days+n_test_days-overlap_days)
prob_confirm_check_train = train_data.ConfirmedCases.value_counts(normalize=True)

prob_fatal_check_train = train_data.Fatalities.value_counts(normalize=True)



n_confirm_train = train_data.ConfirmedCases.value_counts()[1:].sum()

n_fatal_train = train_data.Fatalities.value_counts()[1:].sum()



print('Percentage of confirmed case records = {0:<2.0f}/{1:<2.0f} = {2:<2.1f}%'.format(n_confirm_train, n_train, prob_confirm_check_train[1:].sum()*100))

print('Percentage of fatality records = {0:<2.0f}/{1:<2.0f} = {2:<2.1f}%'.format(n_fatal_train, n_train, prob_fatal_check_train[1:].sum()*100))
#train_data_by_country = train_data.groupby(['Country_Region'],as_index=True).agg({'ConfirmedCases': 'max', 'Fatalities': 'max'})

#train_data_by_country_confirm = train_data_by_country.sort_values(by=["ConfirmedCases"], ascending=False)

train_data_by_country = train_data.groupby(['Date','Country_Region'],as_index=False).agg({'ConfirmedCases': 'sum', 'Fatalities': 'sum',

                                                                                         'GrowthRate':'mean' })

max_train_date = train_data['Date'].max()

train_data_by_country_confirm = train_data_by_country.query('(Date == @max_train_date) & (ConfirmedCases > 100)').sort_values('ConfirmedCases', ascending=False)

train_data_by_country_confirm.set_index('Country_Region', inplace=True)

display(train_data_by_country_confirm.head())



from itertools import cycle, islice

discrete_col = list(islice(cycle(['blue', 'r', 'g', 'k', 'b', 'c', 'm']), None, len(train_data_by_country_confirm.head(30))))

plt.rcParams.update({'font.size': 22})

train_data_by_country_confirm.head(20).plot(figsize=(20,15), kind='barh', color=discrete_col)

plt.legend(["Confirmed Cases", "Fatalities"]);

plt.xlabel("Covid-19 Affected")

plt.title("First 30 Countries with Highest Confirmed Cases")

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



train_data['Date'] = pd.to_datetime(train_data['Date'])

train_data_by_date = train_data.groupby(['Date'],as_index=True).agg({'ConfirmedCases': 'sum','Fatalities': 'sum', 

                                                                     'NewConfirmedCases':'sum', 'NewFatalities':'sum', 'MortalityRate':'mean'})

num0 = train_data_by_date._get_numeric_data() 

num0[num0 < 0.0] = 0.0

#display(train_data_by_date.head())



## ======= Sort by countries with fatalities > 500 ========



train_data_by_country_max = train_data.groupby(['Country_Region'],as_index=True).agg({'ConfirmedCases': 'max', 'Fatalities': 'max'})

train_data_by_country_fatal = train_data_by_country_max[train_data_by_country_max['Fatalities']>500]

train_data_by_country_fatal = train_data_by_country_fatal.sort_values(by=['Fatalities'],ascending=False).reset_index()

display(train_data_by_country_fatal.head(20))



df_merge_by_country = pd.merge(train_data,train_data_by_country_fatal['Country_Region'],on=['Country_Region'],how='inner')

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

fig,(ax4,ax5) = plt.subplots(1,2,figsize=(20, 8))

#train_data_by_date.loc[(train_data_by_date.ConfirmedCases > 200)]#useless, its already summed.

train_data_by_date.MortalityRate.plot(ax=ax4, x_compat=True, legend='Mortality Rate',color='g')#tell pandas not to use its own datetime format

reformat_time(0,ax4)

for num, country in enumerate(countries):

    match = df_max_fatality_country.Country_Region==country 

    df_fatality_by_country = df_max_fatality_country[match] 

    df_fatality_by_country.MortalityRate.plot(ax=ax5, x_compat=True, title='Average Mortality Rate By Country')    

    reformat_time(0,ax5)



ax5.legend(countries, loc='center left',bbox_to_anchor=(1.0, 0.5))
train_data_by_max_date = train_data_by_country.query('(Date == @max_train_date) & (ConfirmedCases > 100)')

train_data_by_max_date.loc[:, 'MortalityRate'] = train_data_by_max_date.loc[:,'Fatalities']/train_data_by_max_date.loc[:,'ConfirmedCases']

train_data_by_mortality = train_data_by_max_date.sort_values('MortalityRate', ascending=False)

train_data_by_mortality.set_index('Country_Region', inplace=True)

display(train_data_by_mortality.head())



#palette = plt.get_cmap('gist_rainbow')

palette = plt.get_cmap('OrRd_r')

rainbow_col = [palette(1.*i/20.0) for i in range(20)]



train_data_by_mortality.MortalityRate.head(20).plot(figsize=(15,10), kind='barh', color=rainbow_col)

plt.xlabel("Mortality Rate")

plt.title("Highest Mortality Rate Countries")

ylocs, ylabs = plt.yticks()

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

#world_population = pd.read_csv("/kaggle/input/population-by-country-2020/population_by_country_2020.csv")

#display(world_population.head()) #for next round
from sklearn.linear_model import Ridge

from sklearn.preprocessing import PolynomialFeatures 

from sklearn.pipeline import make_pipeline

from tqdm import tqdm



plt.rcParams.update({'font.size': 12})

fig,(ax0,ax1) = plt.subplots(1,2,figsize=(20, 8))

countries_europe = ['Italy', 'France', 'Spain', 'Germany', 'United Kingdom']



# Take the 1st day as 2020-02-23

df = train_data.loc[train_data.Date >= '2020-02-23']

n_days_europe = df.Date.nunique()

rainbow_col= plt.cm.jet(np.linspace(0,1,len(countries)))



for country, c in tqdm(zip(countries,rainbow_col)): 

    df_country_train = df_max_fatality_country[df_max_fatality_country['Country_Region']==country] 

    df_country_test = test_data[test_data['Country_Region']==country]  

    df_country_train = df_country_train.reset_index()[df_country_train.reset_index().Date > '2020-02-22']

    n_days_sans_China = df.Date.nunique() - df_country_train.Date.nunique() 

    

    x_train = np.arange(1, n_days_europe+1).reshape((-1,1))

    x_test  = (np.arange(1,n_days_europe+n_test_days+1-overlap_days)).reshape((-1,1)) 

    y_train_f = df_country_train['Fatalities']

    #print (x_train, y_train_f)

    model_f = make_pipeline(PolynomialFeatures(degree=3), Ridge(fit_intercept=False)) 

    model_f = model_f.fit(x_train, y_train_f)

    y_predict_f = model_f.predict(x_test) 

    #print (x_test[-n_test_days:], y_predict_f[-n_test_days:])

    y_train_c = df_country_train['ConfirmedCases'] 

    model_c = make_pipeline(PolynomialFeatures(degree=3), Ridge(fit_intercept=False)) 

    model_c = model_c.fit(x_train, y_train_c)

    y_predict_c = model_c.predict(x_test)

    

    extend_days_test = [i+len(x_test) for i in range(n_days_sans_China)]

    x_test      = np.append(x_test, extend_days_test) 

    y_predict_c = np.pad(y_predict_c, (n_days_sans_China, 0), 'constant')

    y_predict_f = np.pad(y_predict_f, (n_days_sans_China, 0), 'constant')

    

    ax0.plot(x_test[-n_test_days:], y_predict_c[-n_test_days:],linewidth=2, label='predict_'+country, color=c)

    ax0.plot(x_train, y_train_c, linewidth=2, color=c, linestyle='dotted', label='train_'+country)

    ax0.set_title("Prediction vs Training for Confirmed Cases")

    ax0.set_xlabel("Number of days")

    ax0.set_ylabel("Confirmed Cases")

    #ax0.legend(loc='center left',bbox_to_anchor=(1.0, 0.5))

    #ax0.set_yscale('log')

    

    ax1.plot(x_test[-(n_test_days):], y_predict_f[-(n_test_days):],linewidth=2, label='predict_'+country, color=c)

    ax1.plot(x_train, y_train_f, linewidth=2, color=c, linestyle='dotted', label='train_'+country)

    ax1.set_title("Prediction vs Training for Fatalities")

    ax1.set_xlabel("Number of days")

    ax1.set_ylabel("Fatalities")

    ax1.legend(loc='center left',bbox_to_anchor=(1.0, 0.5))

    #ax1.set_yscale('log')
%%time

datadir = Path('/kaggle/input/covid19-global-forecasting-week-4')



# Read in the data CSV files

train = pd.read_csv(datadir/'train.csv')

test = pd.read_csv(datadir/'test.csv')

submission = pd.read_csv(datadir/'submission.csv')
train.rename({'Country_Region': 'country', 'Province_State': 'province', 'Id': 'id', 'Date': 'date', 'ConfirmedCases': 'confirmed', 'Fatalities': 'fatalities'}, axis=1, inplace=True)

test.rename({'Country_Region': 'country', 'Province_State': 'province', 'Id': 'id', 'Date': 'date', 'ConfirmedCases': 'confirmed', 'Fatalities': 'fatalities'}, axis=1, inplace=True)

train['country_province'] = train['country'].fillna('') + '/' + train['province'].fillna('')

test['country_province'] = test['country'].fillna('') + '/' + test['province'].fillna('')

ww_df = train.groupby('date')[['confirmed', 'fatalities']].sum().reset_index()

ww_df['new_case'] = ww_df['confirmed'] - ww_df['confirmed'].shift(1)

ww_df.tail()
ww_melt_df = pd.melt(ww_df, id_vars=['date'], value_vars=['confirmed', 'fatalities', 'new_case'])

ww_melt_df
fig = px.line(ww_melt_df, x="value", y="date", color='variable', 

              title="Worldwide Confirmed/Death Cases ")

fig.show()
fig = px.line(ww_melt_df, x="date", y="value", color='variable',

              title="Worldwide Confirmed/Death Cases Over Time (Log scale)",

             log_y=True)

fig.show()
ww_df['mortality'] = ww_df['fatalities'] / ww_df['confirmed']



fig = px.line(ww_df, x="date", y="mortality", 

              title="Worldwide Mortality Rate ")

fig.show()