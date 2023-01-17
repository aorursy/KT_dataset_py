import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings('ignore')

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import requests

from sklearn.preprocessing import OrdinalEncoder

import xgboost as xgb

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_squared_log_error

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

#Load Data

df_train = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv", parse_dates=['Date'])

df_test = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv", parse_dates=['Date'])

# #Change ConfirmedCases and Fatalities datatype to int and Date string to Datetime format

df_train['ConfirmedCases'] = df_train['ConfirmedCases'].astype(int)

df_train['Fatalities'] = df_train['Fatalities'].astype(int)

dfd = df_train.copy()

vmaxdate = dfd['Date'].max().strftime("%b %d %Y")

df_dates = dfd.groupby('Date', as_index=False)['ConfirmedCases','Fatalities'].sum()

df_dates['DailyConf'] = df_dates['ConfirmedCases'].diff()

df_dates['DailyFat'] = df_dates['Fatalities'].diff()

df_dates.dropna(inplace=True)

df_dates['DailyConf'] = df_dates['DailyConf'].astype(int)

df_dates['DailyFat'] = df_dates['DailyFat'].astype(int)

df_dates['CasesR7dayMean'] = df_dates.DailyConf.rolling(7).mean()

df_dates['DeathsR7dayMean'] = df_dates.DailyFat.rolling(7).mean()
fig = make_subplots(rows=1, cols=2)

fig.add_trace(

    go.Scatter(x=df_dates['Date'], y=df_dates['ConfirmedCases'],name='Total Cases'),

    row=1, col=1

)

fig.add_trace(

    go.Scatter(x=df_dates['Date'], y=df_dates['Fatalities'], name = 'Total Deaths'),

    row=1, col=2

)

fig.update_layout(height=450, width=830, title_text="Total Cases and Total Deaths of the World as of " + vmaxdate)

fig.show()

fig = make_subplots(rows=1, cols=2)

fig.add_trace(

    go.Scatter(x=df_dates['Date'], y=np.log(df_dates['ConfirmedCases']),name='Total Cases'),

    row=1, col=1

)

fig.add_trace(

    go.Scatter(x=df_dates['Date'], y=np.log(df_dates['Fatalities']), name = 'Total Deaths'),

    row=1, col=2

)

fig.update_layout(height=450, width=830, title_text="Logarithmic Cases and Deaths of the World as of " + vmaxdate)

fig.show()
fig = make_subplots(rows=2, cols=1, 

                    shared_xaxes=True, 

                    vertical_spacing=0.02)

fig.add_trace(go.Scatter(x=df_dates['Date'], y=df_dates['DailyConf'], name='Daily Cases'),

              row=2, col=1)



fig.add_trace(go.Scatter(x=df_dates['Date'], y=df_dates['DailyFat'], name = 'Daily Deaths'),

              row=1, col=1)



fig.update_layout(height=600, width=800,

                  title_text="Daily Deaths and Cases of COVID-19 till " + vmaxdate)

fig.show()
fig = go.Figure(data=[

    go.Bar(name='Daily Cases', x=df_dates['Date'], y=df_dates['DailyConf']),

    go.Bar(name='Daily Deaths', x=df_dates['Date'], y=df_dates['DailyFat'])

])



fig.add_trace(go.Scatter(name='Cases:7-day rolling average',x=df_dates['Date'],y=df_dates['CasesR7dayMean'],marker_color='black'))

fig.add_trace(go.Scatter(name='Deaths:7-day rolling average',x=df_dates['Date'],y=df_dates['DeathsR7dayMean'],marker_color='yellow'))



# Change the bar mode

fig.update_layout(barmode='overlay', title='Worldwide daily Case and Death count',showlegend=True)

fig.show()
# Dictionary to get the state codes from state names for US

us_state_code = {

    'Alabama': 'AL',

    'Alaska': 'AK',

    'American Samoa': 'AS',

    'Arizona': 'AZ',

    'Arkansas': 'AR',

    'California': 'CA',

    'Colorado': 'CO',

    'Connecticut': 'CT',

    'Delaware': 'DE',

    'District of Columbia': 'DC',

    'Florida': 'FL',

    'Georgia': 'GA',

    'Guam': 'GU',

    'Hawaii': 'HI',

    'Idaho': 'ID',

    'Illinois': 'IL',

    'Indiana': 'IN',

    'Iowa': 'IA',

    'Kansas': 'KS',

    'Kentucky': 'KY',

    'Louisiana': 'LA',

    'Maine': 'ME',

    'Maryland': 'MD',

    'Massachusetts': 'MA',

    'Michigan': 'MI',

    'Minnesota': 'MN',

    'Mississippi': 'MS',

    'Missouri': 'MO',

    'Montana': 'MT',

    'Nebraska': 'NE',

    'Nevada': 'NV',

    'New Hampshire': 'NH',

    'New Jersey': 'NJ',

    'New Mexico': 'NM',

    'New York': 'NY',

    'North Carolina': 'NC',

    'North Dakota': 'ND',

    'Northern Mariana Islands':'MP',

    'Ohio': 'OH',

    'Oklahoma': 'OK',

    'Oregon': 'OR',

    'Pennsylvania': 'PA',

    'Puerto Rico': 'PR',

    'Rhode Island': 'RI',

    'South Carolina': 'SC',

    'South Dakota': 'SD',

    'Tennessee': 'TN',

    'Texas': 'TX',

    'Utah': 'UT',

    'Vermont': 'VT',

    'Virgin Islands': 'VI',

    'Virginia': 'VA',

    'Washington': 'WA',

    'West Virginia': 'WV',

    'Wisconsin': 'WI',

    'Wyoming': 'WY'

}

dfu = df_train.copy()

df_us = dfu[dfu['Country_Region']=='US']

df_us['State_Code'] = df_us.apply(lambda x: us_state_code.get(x.Province_State, float('nan')), axis=1)

df_us['Date'] = df_us['Date'].astype(str)

df_us['log(ConfirmedCases)'] = np.log(df_us.ConfirmedCases + 1)

df_us['log(Fatalities)'] = np.log(df_us.Fatalities + 1)

df_us.head()

px.choropleth(df_us,

              locationmode="USA-states",

              scope="usa",

              locations='State_Code',

              color="log(ConfirmedCases)",

              hover_name="Province_State",

              hover_data=["ConfirmedCases"],

              animation_frame="Date",

              color_continuous_scale=px.colors.sequential.Reds,

              title = 'Total Cases growth for USA(Logarithmic Scale)')
dft = df_train.copy()

df_totals = dft[dft['Date'] == np.max(dft['Date'])].groupby('Country_Region', as_index=False)['ConfirmedCases','Fatalities'].sum().sort_values('ConfirmedCases', ascending= False)

#df.head()Province_State	Country_Region	Date	ConfirmedCases	FatalitiesProvince_State	Country_Region	Date	ConfirmedCases	Fatalities

df_totals['MortalityRate'] = np.round(((df_totals['Fatalities']/df_totals['ConfirmedCases'])*100),1)

df_top15 = df_totals.head(15)

df_top15.reset_index(drop=True, inplace=True)

df_top15.style.background_gradient(cmap='viridis')
fig = px.parallel_coordinates(df_top15, color = df_top15.index )

                              #color_continuous_scale=px.colors.diverging.Tealrose, color_continuous_midpoint=2)

#fig.update_layout(title='Worldwide daily Case and Death count')

fig.show()
dfs = df_train.copy()

df_totalss = dft[dft['Date'] == np.max(dft['Date'])].groupby(['Country_Region','Province_State'], as_index=False)['ConfirmedCases','Fatalities'].sum().sort_values('ConfirmedCases', ascending= False)

#df.head()Province_State	Country_Region	Date	ConfirmedCases	FatalitiesProvince_State	Country_Region	Date	ConfirmedCases	Fatalities

df_totalss['MortalityRate'] = np.round(((df_totalss['Fatalities']/df_totalss['ConfirmedCases'])*100),1)

df_top15s = df_totalss.head(15)

df_top15s.reset_index(drop=True, inplace=True)

df_top15s.style.background_gradient(cmap='viridis')
dfss= df_top15s.head(11)

fig = px.scatter(dfss, x= np.log(dfss['ConfirmedCases']), y=np.log(dfss['Fatalities']), size="MortalityRate", color="Country_Region",

           hover_name="Province_State", log_x=True, size_max=60)

fig.update_layout(barmode='overlay', title='Top 11 States/Provinces with highest cases ',

                  title_font_size=18, xaxis_title="Log Cases",

                  yaxis_title="Log Deaths",showlegend=True)

fig.show()
dfc = df_train.copy()

df_totalc = dfc[(dfc['Date'] == np.max(dfc['Date'])) & (dfc['Country_Region'] == 'Canada')].groupby(['Province_State'], as_index=False)['ConfirmedCases','Fatalities'].sum().sort_values('ConfirmedCases', ascending= False)

#df.head()Province_State	Country_Region	Date	ConfirmedCases	FatalitiesProvince_State	Country_Region	Date	ConfirmedCases	Fatalities

df_totalc['MortalityRate'] = np.round(((df_totalc['Fatalities']/df_totalc['ConfirmedCases'])*100),1)

#df_top15s = df_totalss.head(15)

df_totalc.reset_index(drop=True, inplace=True)

df_totalc.style.background_gradient(cmap='viridis')
fig = px.scatter(df_totalc, x= 'ConfirmedCases', y='Fatalities', size="MortalityRate", color="Province_State",

        hover_name="Province_State", size_max=30)

fig.update_layout( title='Canadian Provinces Mortality Rate ',

                  title_font_size=18, xaxis_title="Cases",

                  yaxis_title="Deaths",showlegend=True)

fig.show()
#Feature Engineering

#Ordinal encoding of Province_State, Country_Region

df = df_train.copy()

dft = df_test.copy()

df['Province_State'].fillna('NaN',inplace=True)

dft['Province_State'].fillna('NaN',inplace=True)

oe = OrdinalEncoder()

df[['Province_State','Country_Region']] = oe.fit_transform(df.loc[:,['Province_State','Country_Region']])

dft[['Province_State','Country_Region']] = oe.fit_transform(dft.loc[:,['Province_State','Country_Region']])

test_date_min = dft['Date'].min()

def create_features(df):

    df['day'] = df['Date'].dt.day

    df['month'] = df['Date'].dt.month

    df['dayofweek'] = df['Date'].dt.dayofweek

    df['dayofyear'] = df['Date'].dt.dayofyear

    df['weekofyear'] = df['Date'].dt.weekofyear

    return df

def avoid_data_leakage(df, date=test_date_min):

    return df[df['Date']<date]

df = create_features(df)

df = avoid_data_leakage(df)

dft = create_features(dft)

train_columns = ['Province_State','Country_Region','day','month','dayofweek','dayofyear','weekofyear', 

            'ConfirmedCases','Fatalities']

dfx= df[train_columns]

test_columns = ['Province_State','Country_Region','day','month','dayofweek','dayofyear','weekofyear']
dftf = dft[test_columns]

X ,yc, yf = dfx.iloc[:,:-2], dfx.iloc[:,-2],dfx.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, yc, test_size=0.2, random_state=7)

xg_regc = xgb.XGBRegressor(objective ='reg:squarederror',eta = 0.2, n_estimators = 1500)

xg_regc.fit(X_train,y_train)

predc = xg_regc.predict(X_test)

predc[predc < 0] = 0

np.around(predc)

#predc.astype(int)

rmsec = np.sqrt(mean_squared_error(y_test,predc))

#print("RMSE: %f" %(rmsec))

lrmsec = np.sqrt(mean_squared_log_error(y_test,predc))

#print("LRMSE: %f" %(lrmsec))

X_train, X_test, y_train, y_test = train_test_split(X, yf, test_size=0.2, random_state=7)

xg_regf = xgb.XGBRegressor(objective ='reg:squarederror',eta = 0.3,n_estimators = 1500)

xg_regf.fit(X_train,y_train)

predf = xg_regf.predict(X_test)

predf[predf < 0] = 0

rmsef = np.sqrt(mean_squared_error(y_test,predf))

#print("RMSE: %f" %(rmsef))

lrmsef = np.sqrt(mean_squared_log_error(y_test,predf))

#print("LRMSE: %f" %(lrmsef))

xgb.plot_importance(xg_regc)

plt.rcParams['figure.figsize'] = [5, 5]

plt.show()

xgb.plot_importance(xg_regf)

plt.rcParams['figure.figsize'] = [5, 5]

plt.show()
submission = []

#Loop through all the unique countries

for country in dfx.Country_Region.unique():

    #Filter on the basis of country

    df_tr = dfx[dfx["Country_Region"]==country]

    #Loop through all the States of the selected country

    for state in df_tr.Province_State.unique():

        #Filter on the basis of state

        df_tr2 = df_tr[df_tr["Province_State"]==state]

        #Convert to numpy array for training

        train = df_tr2.values

        #Separate the features and labels

        X_train, y_train = train[:,:-2], train[:,-2:]

        #model1 for predicting Confirmed Cases

        modelc = xgb.XGBRegressor(objective ='reg:squarederror',eta = 0.2,n_estimators = 1500)

        modelc.fit(X_train, y_train[:,0])

        #model2 for predicting Fatalities

        modelf = xgb.XGBRegressor(objective ='reg:squarederror',eta = 0.3, n_estimators = 1500)

        modelf.fit(X_train, y_train[:,1])

        #Get the test data for that particular country and state

        df_test1 = dft[(dft["Country_Region"]==country) & (dft["Province_State"] == state)]

        #Store the ForecastId separately

        ForecastId = df_test1.ForecastId.values

        #Remove the unwanted columns

        df_test2 = df_test1[test_columns]

        y_pred1 = modelc.predict(df_test2.values)

        y_pred2 = modelf.predict(df_test2.values)

        #Append the predicted values to submission list

        for i in range(len(y_pred1)):

            d = {'ForecastId':ForecastId[i], 'ConfirmedCases':y_pred1[i], 'Fatalities':y_pred2[i]}

            submission.append(d)

df_sub = pd.DataFrame(submission)

df_sub.to_csv(r'submission.csv', index=False)
df_forcast = pd.concat([df_test,df_sub.iloc[:,1:]], axis=1)

df_for = df_forcast.copy()

df_for = df_for.groupby('Date',as_index=False)['ConfirmedCases','Fatalities'].sum()

vfmaxdate = df_for['Date'].max().strftime("%b %d %Y")

df_for['DailyConf'] = df_dates['ConfirmedCases'].diff()

df_for['DailyFat'] = df_dates['Fatalities'].diff()

df_for.dropna(inplace=True)

df_for['DailyConf'] = df_for['DailyConf'].astype(int)

df_for['DailyFat'] = df_for['DailyFat'].astype(int)

fig = make_subplots(rows=2, cols=1, 

                    shared_xaxes=True, 

                    vertical_spacing=0.02)

fig.add_trace(go.Scatter(x=df_for['Date'], y=df_for['DailyConf'], name='Daily Cases'),

              row=2, col=1)



fig.add_trace(go.Scatter(x=df_for['Date'], y=df_for['DailyFat'], name = 'Daily Deaths'),

              row=1, col=1)



fig.update_layout(height=600, width=800,

                  title_text="Daily Deaths and Cases of COVID-19 till " + vfmaxdate)

fig.show()