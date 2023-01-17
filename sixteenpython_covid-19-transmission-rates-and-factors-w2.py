# This Python 3 environment comes with many helpful analytics libraries installed

# Loading datasets required for analysis



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)

import warnings # current version of seaborn generates a bunch of warnings that we'll ignore

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))
full_table = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv', 

                         parse_dates=['Date'])

full_table.head()
# Defining COVID-19 cases as per classifications 

cases = ['Confirmed', 'Deaths', 'Recovered', 'Active']



# Defining Active Case: Active Case = confirmed - deaths - recovered

full_table['Active'] = full_table['Confirmed'] - full_table['Deaths'] - full_table['Recovered']



# Renaming Mainland china as China in the data table

full_table['Country/Region'] = full_table['Country/Region'].replace('Mainland China', 'China')



# filling missing values 

full_table[['Province/State']] = full_table[['Province/State']].fillna('')

full_table[cases] = full_table[cases].fillna(0)



# cases in the ships

ship = full_table[full_table['Province/State'].str.contains('Grand Princess')|full_table['Country/Region'].str.contains('Cruise Ship')]



# china and the row

china = full_table[full_table['Country/Region']=='China']

row = full_table[full_table['Country/Region']!='China']



# latest

full_latest = full_table[full_table['Date'] == max(full_table['Date'])].reset_index()

china_latest = full_latest[full_latest['Country/Region']=='China']

row_latest = full_latest[full_latest['Country/Region']!='China']



# latest condensed

full_latest_grouped = full_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

china_latest_grouped = china_latest.groupby('Province/State')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

row_latest_grouped = row_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
temp = full_table.groupby(['Country/Region', 'Province/State'])['Confirmed', 'Deaths', 'Recovered', 'Active'].max()
temp = full_table.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

temp = temp[temp['Date']==max(temp['Date'])].reset_index(drop=True)

temp.style.background_gradient(cmap='Pastel1')
temp_f = full_latest_grouped.sort_values(by='Confirmed', ascending=False)

temp_f = temp_f.reset_index(drop=True)

temp_f.head(11).style.background_gradient(cmap='Reds')
import plotly as py

import plotly.graph_objects as go

import pandas as pd

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)    #THIS LINE IS MOST IMPORTANT AS THIS WILL DISPLAY PLOT ON 

#NOTEBOOK WHILE KERNEL IS RUNNING



#Time Series plot for knwoing the spread



fig = go.Figure()

fig.add_trace(go.Scatter(

                x=full_table.Date,

                y=full_table['Confirmed'],

                name="Confirmed",

                line_color='deepskyblue',

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=full_table.Date,

                y=full_table['Recovered'],

                name="Recovered",

                line_color='dimgray',

                opacity=0.8))

fig.update_layout(title_text='Time Series with Rangeslider',

                  xaxis_rangeslider_visible=True)

py.offline.iplot(fig)
import plotly.offline as py

py.init_notebook_mode(connected=True)



# Calculating the count of confirmed cases by country



countries = np.unique(temp_f['Country/Region'])

mean_conf = []

for country in countries:

    mean_conf.append(temp_f[temp_f['Country/Region'] == country]['Confirmed'].sum())

    

# Building the dataframe



    data = [ dict(

        type = 'choropleth',

        locations = countries,

        z = mean_conf,

        locationmode = 'country names',

        text = countries,

        marker = dict(

            line = dict(color = 'rgb(0,0,0)', width = 1)),

            colorbar = dict(autotick = True, tickprefix = '', 

            title = 'Count')

            )

       ]

    

# Building the visual



    layout = dict(

    title = 'COVID-19 Confirmed Cases',

    geo = dict(

        showframe = False,

        showocean = True,

        oceancolor = 'rgb(0,255,255)',

        projection = dict(

        type = 'orthographic',

            rotation = dict(

                    lon = 60,

                    lat = 10),

        ),

        lonaxis =  dict(

                showgrid = True,

                gridcolor = 'rgb(102, 102, 102)'

            ),

        lataxis = dict(

                showgrid = True,

                gridcolor = 'rgb(102, 102, 102)'

                )

            ),

        )



fig = dict(data=data, layout=layout)

py.iplot(fig, validate=False, filename='worldmap')
import pandas as pd

global_temp_country = pd.read_csv("../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCountry.csv")
global_temp_country.head()
import plotly.offline as py

py.init_notebook_mode(connected=True)



## Removing the duplicates



global_temp_country_clear = global_temp_country[~global_temp_country['Country'].isin(

    ['Denmark', 'Antarctica', 'France', 'Europe', 'Netherlands',

     'United Kingdom', 'Africa', 'South America'])]



global_temp_country_clear = global_temp_country_clear.replace(

   ['Denmark (Europe)', 'France (Europe)', 'Netherlands (Europe)', 'United Kingdom (Europe)'],

   ['Denmark', 'France', 'Netherlands', 'United Kingdom'])



#Calculating average temperature by country



countries = np.unique(global_temp_country_clear['Country'])

mean_temp = []

for country in countries:

    mean_temp.append(global_temp_country_clear[global_temp_country_clear['Country'] == 

                                               country]['AverageTemperature'].mean())



# Building the data frame

    

data = [ dict(

        type = 'choropleth',

        locations = countries,

        z = mean_temp,

        locationmode = 'country names',

        text = countries,

        marker = dict(

            line = dict(color = 'rgb(0,0,0)', width = 1)),

            colorbar = dict(autotick = True, tickprefix = '', 

            title = '# Average\nTemperature,\n°C')

            )

       ]



# Building the visual



layout = dict(

    title = 'GLOBAL AVERAGE LAND TEMPERATURES',

    geo = dict(

        showframe = False,

        showocean = True,

        oceancolor = 'rgb(0,255,255)',

        projection = dict(

        type = 'orthographic',

            rotation = dict(

                    lon = 60,

                    lat = 10),

        ),

        lonaxis =  dict(

                showgrid = True,

                gridcolor = 'rgb(102, 102, 102)'

            ),

        lataxis = dict(

                showgrid = True,

                gridcolor = 'rgb(102, 102, 102)'

                )

            ),

        )



fig = dict(data=data, layout=layout)

py.iplot(fig, validate=False, filename='worldmap')
import plotly.express as px

import plotly.offline as py

py.init_notebook_mode(connected=True)

formated_gdf = full_table.groupby(['Date', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered'].max()

formated_gdf = formated_gdf.reset_index()

formated_gdf['Date'] = pd.to_datetime(formated_gdf['Date'])

formated_gdf['Date'] = formated_gdf['Date'].dt.strftime('%m/%d/%Y')

formated_gdf['size'] = formated_gdf['Confirmed'].pow(0.3)



fig = px.scatter_geo(formated_gdf, locations="Country/Region", locationmode='country names', 

                     color="Confirmed", size='size', hover_name="Country/Region", 

                     range_color= [0, max(formated_gdf['Confirmed'])+2], 

                     projection="natural earth", animation_frame="Date", 

                     title='Progression of spread of COVID-19')

fig.update(layout_coloraxis_showscale=False)

py.offline.iplot(fig)
import warnings



from scipy.optimize import curve_fit

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from tqdm.notebook import tqdm
plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = [20, 8]



warnings.filterwarnings('ignore')
train = pd.read_csv('../input/covid19-global-forecasting-week-2/train.csv',

    parse_dates=['Date'])

test = pd.read_csv('../input/covid19-global-forecasting-week-2/test.csv',

    parse_dates=['Date'])

submission = pd.read_csv('../input/covid19-global-forecasting-week-2/submission.csv',

    index_col='ForecastId')
train.head()
test.head()
temptrain = train.groupby(['Country_Region'])['Id', 'ConfirmedCases', 'Fatalities'].max()
temptrain.head()
temptest = train.groupby(['Country_Region'])['Id' ,'ConfirmedCases', 'Fatalities'].max()
temptest.head()
train.rename(columns={'Country_Region':'Country'}, inplace=True)

test.rename(columns={'Country_Region':'Country'}, inplace=True)



train.rename(columns={'Province_State':'State'}, inplace=True)

test.rename(columns={'Province_State':'State'}, inplace=True)
train['Date'] = pd.to_datetime(train['Date'], infer_datetime_format=True)

test['Date'] = pd.to_datetime(test['Date'], infer_datetime_format=True)
y1_Train = train.iloc[:, -2]

y1_Train.head()
y2_Train = train.iloc[:, -1]

y2_Train.head()
EMPTY_VAL = "EMPTY_VAL"

def fillState(state, country):

    if state == EMPTY_VAL: return country

    return state
#X_Train = df_train.loc[:, ['State', 'Country', 'Date']]

X_Train = train.copy()



X_Train['State'].fillna(EMPTY_VAL, inplace=True)

X_Train['State'] = X_Train.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)



X_Train.loc[:, 'Date'] = X_Train.Date.dt.strftime("%m%d")

X_Train["Date"]  = X_Train["Date"].astype(int)



X_Train.head()
X_Test = test.copy()



X_Test['State'].fillna(EMPTY_VAL, inplace=True)

X_Test['State'] = X_Test.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)



X_Test.loc[:, 'Date'] = X_Test.Date.dt.strftime("%m%d")

X_Test["Date"]  = X_Test["Date"].astype(int)



X_Test.head()
from sklearn import preprocessing



le = preprocessing.LabelEncoder()
X_Train.Country = le.fit_transform(X_Train.Country)

X_Train['State'] = le.fit_transform(X_Train['State'])



X_Train.head()
X_Test.Country = le.fit_transform(X_Test.Country)

X_Test['State'] = le.fit_transform(X_Test['State'])



X_Test.head()
from warnings import filterwarnings

filterwarnings('ignore')
le = preprocessing.LabelEncoder()
from xgboost import XGBRegressor

import lightgbm as lgb
countries = X_Train.Country.unique()



df_out = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})

df_out2 = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})



for country in countries:

    states = X_Train.loc[X_Train.Country == country, :].State.unique()

    #print(country, states)

    # check whether string is nan or not

    for state in states:

        X_Train_CS = X_Train.loc[(X_Train.Country == country) & (X_Train.State == state), ['State', 'Country', 'Date', 'ConfirmedCases', 'Fatalities']]

        

        y1_Train_CS = X_Train_CS.loc[:, 'ConfirmedCases']

        y2_Train_CS = X_Train_CS.loc[:, 'Fatalities']

        

        X_Train_CS = X_Train_CS.loc[:, ['State', 'Country', 'Date']]

        

        X_Train_CS.Country = le.fit_transform(X_Train_CS.Country)

        X_Train_CS['State'] = le.fit_transform(X_Train_CS['State'])

        

        X_Test_CS = X_Test.loc[(X_Test.Country == country) & (X_Test.State == state), ['State', 'Country', 'Date', 'ForecastId']]

        

        X_Test_CS_Id = X_Test_CS.loc[:, 'ForecastId']

        X_Test_CS = X_Test_CS.loc[:, ['State', 'Country', 'Date']]

        

        X_Test_CS.Country = le.fit_transform(X_Test_CS.Country)

        X_Test_CS['State'] = le.fit_transform(X_Test_CS['State'])

        

        # XGBoost

        model1 = XGBRegressor(n_estimators=2000)

        model1.fit(X_Train_CS, y1_Train_CS)

        y1_pred = model1.predict(X_Test_CS)

        

        model2 = XGBRegressor(n_estimators=2000)

        model2.fit(X_Train_CS, y2_Train_CS)

        y2_pred = model2.predict(X_Test_CS)

        

        # LightGBM

        model3 = lgb.LGBMRegressor(n_estimators=2000)

        model3.fit(X_Train_CS, y1_Train_CS)

        y3_pred = model3.predict(X_Test_CS)

        

        model4 = lgb.LGBMRegressor(n_estimators=2000)

        model4.fit(X_Train_CS, y2_Train_CS)

        y4_pred = model4.predict(X_Test_CS)

        

        df = pd.DataFrame({'ForecastId': X_Test_CS_Id, 'ConfirmedCases': y1_pred, 'Fatalities': y2_pred})

        df2 = pd.DataFrame({'ForecastId': X_Test_CS_Id, 'ConfirmedCases': y3_pred, 'Fatalities': y4_pred})

        df_out = pd.concat([df_out, df], axis=0)

        df_out2 = pd.concat([df_out2, df2], axis=0)

    # Done for state loop

# Done for country Loop
df_out.ForecastId = df_out.ForecastId.astype('int')

df_out2.ForecastId = df_out2.ForecastId.astype('int')

df_out['ConfirmedCases'] = (1/2)*(df_out['ConfirmedCases'] + df_out2['ConfirmedCases'])

df_out['Fatalities'] = (1/2)*(df_out['Fatalities'] + df_out2['Fatalities'])

df_out['ConfirmedCases'] = df_out['ConfirmedCases'].round().astype(int)

df_out['Fatalities'] = df_out['Fatalities'].round().astype(int)

df_out.tail()
df_out.to_csv('submission.csv', index=False)