# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.offline as py

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import datetime as dt

from itertools import cycle, islice

py.init_notebook_mode(connected=True)

import matplotlib.dates as dates

import plotly.express as px

import plotly.graph_objects as go



from itertools import cycle, islice

from sklearn.linear_model import LinearRegression

import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
pw4 = '/kaggle/input/covid19-global-forecasting-week-4'



df_Train = pd.read_csv(f'{pw4}/train.csv', parse_dates=["Date"], engine='python')

df_Test = pd.read_csv(f'{pw4}/test.csv', parse_dates=["Date"], engine='python')
train_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")

display(train_data.head())

test_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")

display(test_data.head())

df_sub=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/submission.csv")
train_df = df_Train[['Date','ConfirmedCases','Fatalities','Country_Region']]

train_df.head()

test_d=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

test_d
train_df.tail()
train_d=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

train_d
sum_of_data = pd.pivot_table(train_data, values=['ConfirmedCases','Fatalities'], index=['Date'],aggfunc=np.sum)

display(sum_of_data.max())
def fillState(state, country):

    if state == EMPTY_VAL: return country

    return state
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
def ColumnInfo(df):

    n_province =  df['Province_State'].nunique()

    n_country  =  df['Country_Region'].nunique()

    n_days     =  df['Date'].nunique()

    start_date =  df['Date'].unique()[0]

    end_date   =  df['Date'].unique()[-1]

    return n_province, n_country, n_days, start_date, end_date



n_train = train_data.shape[0]

n_test = test_data.shape[0]



n_prov_train, n_count_train, n_train_days, start_date_train, end_date_train = ColumnInfo(train_data)

n_prov_test,  n_count_test,  n_test_days,  start_date_test,  end_date_test  = ColumnInfo(test_data)



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
df_Train.rename(columns={'Country_Region':'Country'}, inplace=True)

df_Test.rename(columns={'Country_Region':'Country'}, inplace=True)



EMPTY_VAL = "EMPTY_VAL"



df_Train.rename(columns={'Province_State':'State'}, inplace=True)

df_Train['State'].fillna(EMPTY_VAL, inplace=True)

df_Train['State'] = df_Train.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)



df_Test.rename(columns={'Province_State':'State'}, inplace=True)

df_Test['State'].fillna(EMPTY_VAL, inplace=True)

df_Test['State'] = df_Test.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)
df_groupByCountry = df_Train.loc[:, ['Country', 'State', 'ConfirmedCases', 'Fatalities']].groupby(['Country', 'State']).max().reset_index().groupby('Country').sum().sort_values(by='ConfirmedCases', ascending=False).reset_index()

df_groupByCountry[:15].style.background_gradient(cmap='viridis_r')
train_data_by_country = train_data.groupby(['Date','Country_Region'],as_index=False).agg({'ConfirmedCases': 'sum', 'Fatalities': 'sum',

                                                                                         'GrowthRate':'last' })

#display(train_data_by_country.tail(10))

max_train_date = train_data['Date'].max()

train_data_by_country_confirm = train_data_by_country.query('(Date == @max_train_date) & (ConfirmedCases > 100)').sort_values('ConfirmedCases', ascending=False)

train_data_by_country_confirm.set_index('Country_Region', inplace=True)



train_data_by_country_confirm.style.background_gradient(cmap='PuBu_r').format({'ConfirmedCases': "{:.0f}", 'GrowthRate': "{:.2f}"})
import plotly.express as px



countries = df_groupByCountry.Country.unique().tolist()

df_plot = df_Train.loc[(df_Train.Country.isin(countries[:10])) & (df_Train.Date >= '2020-03-11'), ['Date', 'Country', 'State', 'ConfirmedCases', 'Fatalities']].groupby(['Date', 'Country', 'State']).max().reset_index().groupby(['Date', 'Country']).sum().sort_values(by='ConfirmedCases', ascending=False).reset_index()



fig = px.bar(df_plot, x="Date", y="ConfirmedCases", color="Country", barmode="stack")

fig.update_layout(title='Rise of Confirmed Cases around top 10 countries')

fig.show()
discrete_col = list(islice(cycle(['purple', 'r', 'g', 'k', 'b', 'c', 'm']), None, len(train_data_by_country_confirm.head(30))))

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

#display(train_data_by_country_fatal.head(20))



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

    df_fatality_by_country.ConfirmedCases.plot(ax=ax1, x_compat=True, title='Confirmed Cases Nationally')

    reformat_time(0,ax1)

    df_fatality_by_country.Fatalities.plot(ax=ax3, x_compat=True, title='Fatalities Nationally')

    reformat_time(0,ax3)

    

ax1.legend(countries)

ax3.legend(countries)

df_Train.loc[: , ['Country', 'State', 'ConfirmedCases', 'Fatalities']].groupby(['Country', 'State']).max().reset_index().nlargest(15, "ConfirmedCases").style.background_gradient(cmap='nipy_spectral')
import plotly.express as px



countries = df_groupByCountry.Country.unique().tolist()

df_plot = df_Train.loc[df_Train.Country.isin(countries[:10]), ['Date', 'Country', 'ConfirmedCases']].groupby(['Date', 'Country']).max().reset_index()



fig = px.line(df_plot, x="Date", y="ConfirmedCases", color='Country')

fig.update_layout(title='No.of Confirmed Cases per Day for Top 10 Countries',

                   xaxis_title='Date',

                   yaxis_title='No.of Confirmed Cases')



fig.show()
from fbprophet import Prophet

def fit_model(data_,interval_width_=0.95,periods_=10):

    data_.columns = ['ds', 'y']

    data_['ds'] = pd.to_datetime(data_['ds'])

    

    model = Prophet(interval_width=interval_width_)

    model.fit(data_)  

    return model



def predict(model,data_):

    data_=data_.rename(columns={'Date':'ds'})

    forecast = model.predict(data_)

    return forecast



def forecast_state(training_data,testing_data,state_name,interval_width=0.95):

    train_confirmed = training_data.groupby('Date').sum()['ConfirmedCases'].reset_index().copy()

    train_fatalities = training_data.groupby('Date').sum()['Fatalities'].reset_index().copy()

    

    model_confirmed=fit_model(train_confirmed)

    confirmed_predictions = predict(model_confirmed,testing_data[['Date']].copy())

    testing_data['ConfirmedCases']=confirmed_predictions['yhat'].astype(np.uint64).tolist()



    model_fatalities=fit_model(train_fatalities)

    fatalities_predictions = predict(model_fatalities,testing_data[['Date']].copy())

    testing_data['Fatalities']=fatalities_predictions['yhat'].astype(np.uint64).tolist()



#     model_fatalities.plot(fatalities_predictions)    

#     model_confirmed.plot(confirmed_predictions)

    return testing_data
EMPTY_VAL = "EMPTY_VAL"



def fillState(Province_State, Country_Region):

    if Province_State == EMPTY_VAL: return Country_Region

    return Province_State



X_train=train_d.copy()

X_test=test_d.copy()

X_train['Province_State'].fillna(EMPTY_VAL, inplace=True)

X_test['Province_State'].fillna(EMPTY_VAL, inplace=True)

X_train['Province_State'] = X_train.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)

X_test['Province_State'] = X_test.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)
world_output=test_d.copy()

world_output['ConfirmedCases']=int(0)

world_output['Fatalities']=int(0)

count=0

total=X_train['Country_Region'].nunique()

for country,grp_country in X_train.groupby(['Country_Region']):

    country_output={}

    for state,grp_state in grp_country.groupby(['Province_State']):

        print(f'{count}/{total} : {country}\t{state}')

        state_test=X_test.loc[X_test.Province_State == state].copy()

        output=forecast_state(grp_state,state_test,state,0.95)

        world_output.update(output)

    count+=1

    world_output=world_output.astype({"ForecastId":int,"ConfirmedCases":int,"Fatalities":int})

    world_output[['ForecastId','ConfirmedCases','Fatalities']].to_csv('submission.csv',index=False)