import gc

import os

from pathlib import Path

import random

import sys



from tqdm.notebook import tqdm

import numpy as np

import pandas as pd

import scipy as sp





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
# Input data files are available in the "../input/" directory.

import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     filenames.sort()

#     for filename in filenames:

#         print(os.path.join(dirname, filename))




%%time

import requests



for filename in ['time_series_covid19_confirmed_global.csv',

                 'time_series_covid19_deaths_global.csv',

                 'time_series_covid19_recovered_global.csv',

                 'time_series_covid19_confirmed_US.csv',

                 'time_series_covid19_deaths_US.csv']:

    print(f'Downloading {filename}')

    url = f'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/{filename}'

    myfile = requests.get(url)

    open(filename, 'wb').write(myfile.content)

    

    from datetime import datetime



def _convert_date_str(df):

    try:

        df.columns = list(df.columns[:4]) + [datetime.strptime(d, "%m/%d/%y").date().strftime("%Y-%m-%d") for d in df.columns[4:]]

    except:

        print('_convert_date_str failed with %y, try %Y')

        df.columns = list(df.columns[:4]) + [datetime.strptime(d, "%m/%d/%Y").date().strftime("%Y-%m-%d") for d in df.columns[4:]]





confirmed_global_df = pd.read_csv('time_series_covid19_confirmed_global.csv')

_convert_date_str(confirmed_global_df)



deaths_global_df = pd.read_csv('time_series_covid19_deaths_global.csv')

_convert_date_str(deaths_global_df)



recovered_global_df = pd.read_csv('time_series_covid19_recovered_global.csv')

_convert_date_str(recovered_global_df)
# Filter out problematic data points (The West Bank and Gaza had a negative value, cruise ships were associated with Canada, etc.)

removed_states = "Recovered|Grand Princess|Diamond Princess"

removed_countries = "US|The West Bank and Gaza"



confirmed_global_df.rename(columns={"Province/State": "Province_State", "Country/Region": "Country_Region"}, inplace=True)

deaths_global_df.rename(columns={"Province/State": "Province_State", "Country/Region": "Country_Region"}, inplace=True)

recovered_global_df.rename(columns={"Province/State": "Province_State", "Country/Region": "Country_Region"}, inplace=True)



confirmed_global_df = confirmed_global_df[~confirmed_global_df["Province_State"].replace(np.nan, "nan").str.match(removed_states)]

deaths_global_df    = deaths_global_df[~deaths_global_df["Province_State"].replace(np.nan, "nan").str.match(removed_states)]

recovered_global_df = recovered_global_df[~recovered_global_df["Province_State"].replace(np.nan, "nan").str.match(removed_states)]



confirmed_global_df = confirmed_global_df[~confirmed_global_df["Country_Region"].replace(np.nan, "nan").str.match(removed_countries)]

deaths_global_df    = deaths_global_df[~deaths_global_df["Country_Region"].replace(np.nan, "nan").str.match(removed_countries)]

recovered_global_df = recovered_global_df[~recovered_global_df["Country_Region"].replace(np.nan, "nan").str.match(removed_countries)]



confirmed_global_melt_df = confirmed_global_df.melt(

    id_vars=['Country_Region', 'Province_State', 'Lat', 'Long'], value_vars=confirmed_global_df.columns[4:], var_name='Date', value_name='ConfirmedCases')

deaths_global_melt_df = deaths_global_df.melt(

    id_vars=['Country_Region', 'Province_State', 'Lat', 'Long'], value_vars=confirmed_global_df.columns[4:], var_name='Date', value_name='Deaths')

recovered_global_melt_df = deaths_global_df.melt(

    id_vars=['Country_Region', 'Province_State', 'Lat', 'Long'], value_vars=confirmed_global_df.columns[4:], var_name='Date', value_name='Recovered')
train = confirmed_global_melt_df.merge(deaths_global_melt_df, on=['Country_Region', 'Province_State', 'Lat', 'Long', 'Date'])

train = train.merge(recovered_global_melt_df, on=['Country_Region', 'Province_State', 'Lat', 'Long', 'Date'])


confirmed_us_df = pd.read_csv('time_series_covid19_confirmed_US.csv')

deaths_us_df = pd.read_csv('time_series_covid19_deaths_US.csv')



confirmed_us_df.drop(['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Combined_Key'], inplace=True, axis=1)

deaths_us_df.drop(['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Combined_Key', 'Population'], inplace=True, axis=1)



confirmed_us_df.rename({'Long_': 'Long'}, axis=1, inplace=True)

deaths_us_df.rename({'Long_': 'Long'}, axis=1, inplace=True)



_convert_date_str(confirmed_us_df)

_convert_date_str(deaths_us_df)



# clean

confirmed_us_df = confirmed_us_df[~confirmed_us_df.Province_State.str.match("Diamond Princess|Grand Princess|Recovered|Northern Mariana Islands|American Samoa")]

deaths_us_df = deaths_us_df[~deaths_us_df.Province_State.str.match("Diamond Princess|Grand Princess|Recovered|Northern Mariana Islands|American Samoa")]



# --- Aggregate by province state ---

#confirmed_us_df.groupby(['Country_Region', 'Province_State'])

confirmed_us_df = confirmed_us_df.groupby(['Country_Region', 'Province_State']).sum().reset_index()

deaths_us_df = deaths_us_df.groupby(['Country_Region', 'Province_State']).sum().reset_index()



# remove lat, long.

confirmed_us_df.drop(['Lat', 'Long'], inplace=True, axis=1)

deaths_us_df.drop(['Lat', 'Long'], inplace=True, axis=1)



confirmed_us_melt_df = confirmed_us_df.melt(

    id_vars=['Country_Region', 'Province_State'], value_vars=confirmed_us_df.columns[2:], var_name='Date', value_name='ConfirmedCases')

deaths_us_melt_df = deaths_us_df.melt(

    id_vars=['Country_Region', 'Province_State'], value_vars=deaths_us_df.columns[2:], var_name='Date', value_name='Deaths')



train_us = confirmed_us_melt_df.merge(deaths_us_melt_df, on=['Country_Region', 'Province_State', 'Date'])



train = pd.concat([train, train_us], axis=0, sort=False)



train_us.rename({'Country_Region': 'country', 'Province_State': 'province', 'Date': 'date', 'ConfirmedCases': 'confirmed', 'Deaths': 'fatalities'}, axis=1, inplace=True)

train_us['country_province'] = train_us['country'].fillna('') + '/' + train_us['province'].fillna('')
%%time

datadir = Path('/kaggle/input/covid19-global-forecasting-week-4')



# Read in the data CSV files

#train = pd.read_csv(datadir/'train.csv')

#test = pd.read_csv(datadir/'test.csv')

#submission = pd.read_csv(datadir/'submission.csv')

train.rename({'Country_Region': 'country', 'Province_State': 'province', 'Id': 'id', 'Date': 'date', 'ConfirmedCases': 'confirmed', 'Deaths': 'fatalities', 'Recovered': 'recovered'}, axis=1, inplace=True)

train['country_province'] = train['country'].fillna('') + '/' + train['province'].fillna('')



# test.rename({'Country_Region': 'country', 'Province_State': 'province', 'Id': 'id', 'Date': 'date', 'ConfirmedCases': 'confirmed', 'Fatalities': 'fatalities'}, axis=1, inplace=True)

# test['country_province'] = test['country'].fillna('') + '/' + test['province'].fillna('')
ww_df = train.groupby('date')[['confirmed', 'fatalities']].sum().reset_index()

ww_df['new_case'] = ww_df['confirmed'] - ww_df['confirmed'].shift(1)

ww_df.tail()



ww_melt_df = pd.melt(ww_df, id_vars=['date'], value_vars=['confirmed', 'fatalities', 'new_case'])

ww_melt_df



fig = px.line(ww_melt_df, x="date", y="value", color='variable', 

              title="Worldwide Confirmed/Death Cases Over Time")

fig.show()



fig = px.line(ww_melt_df, x="date", y="value", color='variable',

              title="Worldwide Confirmed/Death Cases Over Time (Log scale)",

             log_y=True)

fig.show()



ww_df['mortality'] = ww_df['fatalities'] / ww_df['confirmed']



fig = px.line(ww_df, x="date", y="mortality", 

              title="Worldwide Mortality Rate Over Time")

fig.show()



country_df = train.groupby(['date', 'country'])[['confirmed', 'fatalities']].sum().reset_index()

country_df.tail()



countries = country_df['country'].unique()

print(f'{len(countries)} countries are in dataset:\n{countries}')





target_date = country_df['date'].max()



print('Date: ', target_date)

for i in [1, 10, 100, 1000, 10000]:

    n_countries = len(country_df.query('(date == @target_date) & confirmed > @i'))

    print(f'{n_countries} countries have more than {i} confirmed cases')

    

ax = sns.distplot(np.log10(country_df.query('date == "2020-03-27"')['confirmed'] + 1))

ax.set_xlim([0, 6])

ax.set_xticks(np.arange(7))

_ = ax.set_xticklabels(['0', '10', '100', '1k', '10k', '100k'])



top_country_df = country_df.query('(date == @target_date) & (confirmed > 1000)').sort_values('confirmed', ascending=False)

top_country_melt_df = pd.melt(top_country_df, id_vars='country', value_vars=['confirmed', 'fatalities'])



fig = px.bar(top_country_melt_df.iloc[::-1],

             x='value', y='country', color='variable', barmode='group',

             title=f'Confirmed Cases/Deaths on {target_date}', text='value', height=1500, orientation='h')

fig.show()



top30_countries = top_country_df.sort_values('fatalities', ascending=False).iloc[:30]['country'].unique()

top30_countries_df = country_df[country_df['country'].isin(top30_countries)]

fig = px.line(top30_countries_df,

              x='date', y='fatalities', color='country',

              title=f'Fatalities for top 30 country as of {target_date}')

fig.show()
top_country_df = country_df.query('(date == @target_date) & (confirmed > 100)')

top_country_df['mortality_rate'] = top_country_df['fatalities'] / top_country_df['confirmed']

top_country_df = top_country_df.sort_values('mortality_rate', ascending=False)

fig = px.bar(top_country_df[:30].iloc[::-1],

             x='mortality_rate', y='country',

             title=f'Mortality rate HIGH: top 30 countries on {target_date}', text='mortality_rate', height=800, orientation='h')

fig.show()
fig = px.bar(top_country_df[-30:],

             x='mortality_rate', y='country',

             title=f'Mortality rate LOW: top 30 countries on {target_date}', text='mortality_rate', height=800, orientation='h')

fig.show()
all_country_df = country_df.query('date == @target_date')

all_country_df['confirmed_log1p'] = np.log10(all_country_df['confirmed'] + 1)

all_country_df['fatalities_log1p'] = np.log10(all_country_df['fatalities'] + 1)

all_country_df['mortality_rate'] = all_country_df['fatalities'] / all_country_df['confirmed']



fig = px.choropleth(all_country_df, locations="country", 

                    locationmode='country names', color="confirmed_log1p", 

                    hover_name="country", hover_data=["confirmed", 'fatalities', 'mortality_rate'],

                    range_color=[all_country_df['confirmed_log1p'].min(), all_country_df['confirmed_log1p'].max()], 

                    color_continuous_scale="peach", 

                    title='Countries with Confirmed Cases')



# I'd like to update colorbar to show raw values, but this does not work somehow...

# Please let me know if you know how to do this!!

trace1 = list(fig.select_traces())[0]

trace1.colorbar = go.choropleth.ColorBar(

    tickvals=[0, 1, 2, 3, 4, 5],

    ticktext=['1', '10', '100', '1000','10000', '10000'])

fig.show()


fig = px.choropleth(all_country_df, locations="country", 

                    locationmode='country names', color="mortality_rate", 

                    hover_name="country", range_color=[0, 0.12], 

                    color_continuous_scale="peach", 

                    title='Countries with mortality rate')

fig.show()
country_df['prev_confirmed'] = country_df.groupby('country')['confirmed'].shift(1)

country_df['new_case'] = country_df['confirmed'] - country_df['prev_confirmed']

country_df['new_case'].fillna(0, inplace=True)

top30_country_df = country_df[country_df['country'].isin(top30_countries)]



fig = px.line(top30_country_df,

              x='date', y='new_case', color='country',

              title=f'DAILY NEW Confirmed cases world wide')

fig.show()
for country in countries:

    province = train.query('country == @country')['province'].unique()

    if len(province) > 1:       

        print(f'Country {country} has {len(province)} provinces: {province}')

usa_state_code_df = pd.read_csv('/kaggle/input/usstatecode/usa_states2.csv')


train_us['mortality_rate'] = train_us['fatalities'] / train_us['confirmed']



# Convert province column to its 2-char code name,

state_name_to_code = dict(zip(usa_state_code_df['state_name'], usa_state_code_df['state_code']))

train_us['province_code'] = train_us['province'].map(state_name_to_code)



# Only show latest days.

train_us_latest = train_us.query('date == @target_date')
fig = px.choropleth(train_us_latest, locations='province_code', locationmode="USA-states",

                    color='confirmed', scope="usa", hover_data=['province', 'fatalities', 'mortality_rate'],

                    title=f'Confirmed cases in US on {target_date}')

fig.show()

train_us_latest.sort_values('confirmed', ascending=False)

fig = px.choropleth(train_us_latest, locations='province_code', locationmode="USA-states",

                    color='mortality_rate', scope="usa", hover_data=['province', 'fatalities', 'mortality_rate'],

                    title=f'Mortality rate in US on {target_date}')

fig.show()
# Ref: https://www.kaggle.com/abhinand05/covid-19-digging-a-bit-deeper

europe_country_list =list([

    'Austria','Belgium','Bulgaria','Croatia','Cyprus','Czechia','Denmark','Estonia','Finland','France','Germany','Greece','Hungary','Ireland',

    'Italy', 'Latvia','Luxembourg','Lithuania','Malta','Norway','Netherlands','Poland','Portugal','Romania','Slovakia','Slovenia',

    'Spain', 'Sweden', 'United Kingdom', 'Iceland', 'Russia', 'Switzerland', 'Serbia', 'Ukraine', 'Belarus',

    'Albania', 'Bosnia and Herzegovina', 'Kosovo', 'Moldova', 'Montenegro', 'North Macedonia'])



country_df['date'] = pd.to_datetime(country_df['date'])

train_europe = country_df[country_df['country'].isin(europe_country_list)]

#train_europe['date_str'] = pd.to_datetime(train_europe['date'])

train_europe_latest = train_europe.query('date == @target_date')



fig = px.choropleth(train_europe_latest, locations="country", 

                    locationmode='country names', color="confirmed", 

                    hover_name="country", range_color=[1, train_europe_latest['confirmed'].max()], 

                    color_continuous_scale='portland', 

                    title=f'European Countries with Confirmed Cases as of {target_date}', scope='europe', height=800)

fig.show()
country_latest = country_df.query('date == @target_date')



fig = px.choropleth(country_latest, locations="country", 

                    locationmode='country names', color="confirmed", 

                    hover_name="country", range_color=[1, 50000], 

                    color_continuous_scale='portland', 

                    title=f'Asian Countries with Confirmed Cases as of {target_date}', scope='asia', height=800)

fig.show()



top_asian_country_df = country_df[country_df['country'].isin(['China', 'Indonesia', 'Iran', 'Japan', 'Korea, South', 'Malaysia', 'Philippines'])]



fig = px.line(top_asian_country_df,

              x='date', y='new_case', color='country',

              title=f'DAILY NEW Confirmed cases world wide')

fig.show()
def sigmoid(t, M, beta, alpha, offset=0):

    alpha += offset

    return M / (1 + np.exp(-beta * (t - alpha)))



def error(x, y, params):

    M, beta, alpha = params

    y_pred = sigmoid(x, M, beta, alpha)



    # apply weight, latest number is more important than past.

    weight = np.arange(len(y_pred)) ** 2

    loss_mse = np.mean((y_pred - y) ** 2 * weight)

    return loss_mse



def gen_random_color(min_value=0, max_value=256) -> str:

    """Generate random color for plotly"""

    r, g, b = np.random.randint(min_value, max_value, 3)

    return f'rgb({r},{g},{b})'





def fit_sigmoid(exclude_days=0):

    target_country_df_list = []

    pred_df_list = []

    for target_country in top30_countries:

        print('target_country', target_country)

        # --- Train ---

        target_country_df = country_df.query('country == @target_country')



        #train_start_date = target_country_df['date'].min()

        train_start_date = target_country_df.query('confirmed > 1000')['date'].min()

        train_end_date = pd.to_datetime(target_date) - pd.Timedelta(f'{exclude_days} days')

        target_date_df = target_country_df.query('(date >= @train_start_date) & (date <= @train_end_date)')

        if len(target_date_df) <= 7:

            print('WARNING: the data is not enough, use 7 more days...')

            train_start_date -= pd.Timedelta('7 days')

            target_date_df = target_country_df.query('(date >= @train_start_date) & (date <= @train_end_date)')



        confirmed = target_date_df['confirmed'].values

        x = np.arange(len(confirmed))



        lossfun = lambda params: error(x, confirmed, params)

        res = sp.optimize.minimize(lossfun, x0=[np.max(confirmed) * 5, 0.04, 2 * len(confirmed) / 3.], method='nelder-mead')

        M, beta, alpha = res.x

        # sigmoid_models[key] = (M, beta, alpha)

        # np.clip(sigmoid(list(range(len(data), len(data) + steps)), M, beta, alpha), 0, None).astype(int)



        # --- Pred ---

        pred_start_date = target_country_df['date'].min()

        pred_end_date = pd.to_datetime('2020-07-01')

        days = int((pred_end_date - pred_start_date) / pd.Timedelta('1 days'))

        # print('pred start', pred_start_date, 'end', pred_end_date, 'days', days)



        x = np.arange(days)

        offset = (train_start_date - pred_start_date) / pd.Timedelta('1 days')

        print('train_start_date', train_start_date, 'offset', offset, 'params', M, beta, alpha)

        y_pred = sigmoid(x, M, beta, alpha, offset=offset)

        # target_country_df['confirmed_pred'] = y_pred



        all_dates = [pred_start_date + np.timedelta64(x, 'D') for x in range(days)]

        pred_df = pd.DataFrame({

            'date': all_dates,

            'country': target_country,

            'confirmed_pred': y_pred,

        })



        target_country_df_list.append(target_country_df)

        pred_df_list.append(pred_df)

    return target_country_df_list, pred_df_list



def plot_sigmoid_fitting(target_country_df_list, pred_df_list, title=''):

    n_countries = len(top30_countries)



    # --- Plot ---

    fig = go.Figure()



    for i in range(n_countries):

        target_country = top30_countries[i]

        target_country_df = target_country_df_list[i]

        pred_df = pred_df_list[i]

        color = gen_random_color(min_value=20)

        # Prediction

        fig.add_trace(go.Scatter(

            x=pred_df['date'], y=pred_df['confirmed_pred'],

            name=f'{target_country}_pred',

            line=dict(color=color, dash='dash')

        ))



        # Ground truth

        fig.add_trace(go.Scatter(

            x=target_country_df['date'], y=target_country_df['confirmed'],

            mode='markers', name=f'{target_country}_actual',

            line=dict(color=color),

        ))

    fig.update_layout(

        title=title, xaxis_title='Date', yaxis_title='Confirmed cases')

    fig.show()
target_country_df_list, pred_df_list = fit_sigmoid(exclude_days=0)
plot_sigmoid_fitting(target_country_df_list, pred_df_list, title='Sigmoid fitting with all latest data')
target_country_df_list, pred_df_list = fit_sigmoid(exclude_days=7)
plot_sigmoid_fitting(target_country_df_list, pred_df_list, title='Sigmoid fitting without last 7days data')
import seaborn as sns

train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/test.csv')

df1 = train.Population.groupby(train['Country_Region']).max().sort_values(ascending= False)

df10 = pd.DataFrame()

df20 = pd.DataFrame()

df10['population'] = df1.iloc[0:10]

df10['country']= df10.index

df20['population'] = df1.iloc[11:20]

df20['country'] = df20.index



train1= train[train['Target']=='ConfirmedCases']

data1 = pd.DataFrame()

data1['values'] =train1.TargetValue.groupby(train1['Country_Region']).sum().sort_values(ascending= False)

data1['country'] = data1.index

data1.index = np.arange(0,len(data1))

data10 = data1.iloc[0:10,:]

data20 = data1.iloc[11:20,:]



train1= train[train['Target']!='ConfirmedCases']

data1 = pd.DataFrame()

data1['values'] =train1.TargetValue.groupby(train1['Country_Region']).sum().sort_values(ascending= False)

data1['country'] = data1.index

data1.index = np.arange(0,len(data1))

data10 = data1.iloc[0:10,:]

data20 = data1.iloc[11:20,:]



india = train[train['Country_Region']=='India']

india.drop(['County','Province_State'],axis =1,inplace =True)

india.index = np.arange(0,len(india)) #rechanging the index

ind = india[india['Target']=='ConfirmedCases']

ind.index = np.arange(0,len(ind))



list1 = []

for i in range(2,7):

    date = '2020'+'-0'+str(i)+'-01'

    list1.append(ind[ind['Date']<date]['TargetValue'].sum())





list2 =[]

for i in range(len(list1)):

    if i ==0:

        list2.append(list1[i])

    else:

        list2.append(list1[i]-list1[i-1])



labels =['Jan','Feb','Mar','Apr','May']

df = india['TargetValue'].groupby(train['Target']).sum()

labels =[df.index[0],df.index[1]]

sizes = [df[0],df[1]]

explode = (0, 0.2)  





wor = train[train['Target']=='ConfirmedCases']

independent_columns = ['Country_Region','Weight','Target','Date']

dependent_column = ['TargetValue']

X= train[independent_columns]

y = train[dependent_column]

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

X['Target']=le.fit_transform(X['Target'])



train1= train[train['Target']=='ConfirmedCases']

data1 = pd.DataFrame()

data1['values'] =train1.TargetValue.groupby(train1['Country_Region']).sum().sort_values(ascending= False)

data1['country'] = data1.index



k = len(data1['country'])

dict1 = {}

for i in data1['country']:

    dict1[i] = k

    k =k-1



list1=[]

X['encoded_country']=0

for i in X['Country_Region']:

    list1.append(dict1[i])

X['encoded_country'] = list1



X['date_dup'] = pd.to_datetime(X['Date'])



X['month'] = 0

list1=[]

for i in X['date_dup']:

    list1.append(i.month)

X['month'] = list1



X['date'] = 0

list1=[]

for i in X['date_dup']:

    list1.append(i.day)

X['date'] = list1

X.drop(['Country_Region','Date','date_dup'],axis =1,inplace =True)


from sklearn.model_selection import train_test_split as tts

max_range =10



from sklearn.ensemble import RandomForestRegressor as regr

from sklearn.metrics import r2_score



X_train,X_test,y_train,y_test = tts(X,y,test_size =0.3,random_state =7)

model = regr()

model.fit(X_train,y_train)



print(r2_score(y_test,model.predict(X_test)))



test = test[independent_columns]

list1=[]

test['encoded_country']=0

for i in test['Country_Region']:

    list1.append(dict1[i])

test['encoded_country'] = list1



test['date_dup'] = pd.to_datetime(test['Date'])



test['month'] = 0

list1=[]

for i in test['date_dup']:

    list1.append(i.month)

test['month'] = list1



test['date'] = 0

list1=[]

for i in test['date_dup']:

    list1.append(i.day)

test['date'] = list1



test.drop(['Country_Region','Date','date_dup'],axis =1,inplace =True)



le1 =LabelEncoder()

test['Target'] = le1.fit_transform(test['Target'])



pred = model.predict(test)

t =pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/test.csv')

ss = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/submission.csv')

output = pd.DataFrame({'Id': t.ForecastId  , 'TargetValue': pred})

a=output.groupby(['Id'])['TargetValue'].quantile(q=0.05).reset_index()

b=output.groupby(['Id'])['TargetValue'].quantile(q=0.5).reset_index()

c=output.groupby(['Id'])['TargetValue'].quantile(q=0.95).reset_index()





a.columns=['Id','q0.05']

b.columns=['Id','q0.5']

c.columns=['Id','q0.95']

a=pd.concat([a,b['q0.5'],c['q0.95']],1)

a['q0.05']=a['q0.05']

a['q0.5']=a['q0.5']

a['q0.95']=a['q0.95']



sub=pd.melt(a, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])

sub['variable']=sub['variable'].str.replace("q","", regex=False)

sub['ForecastId_Quantile']=sub['Id'].astype(str)+'_'+sub['variable']

sub['TargetValue']=sub['value']

sub=sub[['ForecastId_Quantile','TargetValue']]

sub.reset_index(drop=True,inplace=True)

sub.to_csv("submission.csv",index=False)

sub.info()
