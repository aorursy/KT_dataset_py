!pip install reverse_geocoder
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

import glob

from math import radians, cos, sin, asin, sqrt

import geopandas as gpd

import geopy

from geopy.geocoders import Nominatim

from geopy.extra.rate_limiter import RateLimiter

locator = Nominatim(user_agent="myGeocoder",timeout=20)
import seaborn as sns

import reverse_geocoder as rg

import matplotlib.animation as animation

import datetime

import pycountry

import calendar
xtrain = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv")

xtest = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/test.csv")
xtrain.head(10)
xtrain['Date'].value_counts()
xtrain['ConfirmedCases'] = xtrain['ConfirmedCases'].astype(int)

xtrain['Fatalities'] = xtrain['Fatalities'].astype(int)
xtrain['Modified_Date'] = pd.to_datetime(xtrain['Date'])
xtrain["month"] = xtrain['Modified_Date'].map(lambda x: x.month)
xtrain.head(10)
xtrain.isnull().sum()
import ssl 



try:

    _create_unverified_https_context = ssl._create_unverified_context

except AttributeError:

    pass

else:

    ssl._create_default_https_context = _create_unverified_https_context
xtrain["geom"] = xtrain["Lat"].map(str) + ',' + xtrain["Long"].map(str)
xtrain['new_state'] = xtrain['Province/State']
null_state_df = xtrain[xtrain['Province/State'].isnull()==True]
unique_geom = null_state_df['geom'].unique()
not_null_state_df = xtrain[xtrain['Province/State'].isnull()==False]

print("Shape of the not null states dataframe",not_null_state_df.shape)

print("Shape of the original dataframe",xtrain.shape)

print("Null values of the not null states dataframe",not_null_state_df.isnull().sum())
xtrain.head(10)
not_null_state_df["new_state"]
xtrain['geom'].nunique()
len(unique_geom)
def fillNullProvince(x):

  coordinates = (x['Lat'],x['Long'])

  result =  rg.search(coordinates)

  return result[0].get('name')
xtrain['Province/State'] = xtrain.apply(lambda x:fillNullProvince(x) if pd.isnull(x['Province/State']) else x['Province/State'] ,axis=1)

xtrain['Province/State'].value_counts()
import geopandas as gp
xtrain.to_csv('covid19work.csv', index=False)
worktrain = xtrain = pd.read_csv('covid19work.csv')
worktrain.head(10)
worktrain['month'] = pd.DatetimeIndex(worktrain['Date']).month 

worktrain['month'] = worktrain['month'].apply(lambda x: calendar.month_abbr[x])
confirmed_cases_by_country = worktrain.groupby('Country/Region').sum()[['ConfirmedCases','Fatalities']]

confirmed_cases_by_country.sort_values(by=['ConfirmedCases','Fatalities'],ascending=False,inplace=True)
confirmed_cases_by_country.head(10)
confirmed_cases_by_country['Country'] = confirmed_cases_by_country.index
plt.figure(figsize=(20,5))

plt.subplot(1,2,1)

confirmed_cases_by_country['ConfirmedCases'].head(10).plot(kind='barh',color=(0.9,0.2,0.2,1.0))

plt.xticks(rotation=90)

xlocs, xlabs = plt.xticks()

xlocs=[i+1 for i in range(0,10)]

xlabs=[i/2 for i in range(0,10)]

for i, v in enumerate(confirmed_cases_by_country['ConfirmedCases'].head(10)):

  plt.text(v, xlocs[i]-0.9 , str(v))

plt.xlabel('total number of cases (Normalized)')

plt.title('Top 10 most infected countries')
plt.subplot(1,2,2)

confirmed_cases_by_country['Fatalities'].head(10).plot(kind='barh',color = (0,0.9,.25,1.0))

for i, v in enumerate(confirmed_cases_by_country['Fatalities'].head(10)):

  plt.text(v, xlocs[i]-0.9 , str(v))

plt.xlabel('total number of cases')

plt.title('Top 10 most fatalities countries')

plt.xticks(rotation=90)

plt.show
from bokeh.io import output_notebook, show, output_file

from bokeh.plotting import figure

from bokeh.models import GeoJSONDataSource, ColumnDataSource

from bokeh.models import HoverTool

import json
merged_json = json.loads(worktrain.to_json())

json_data = json.dumps(merged_json)



geosource = GeoJSONDataSource(geojson = json_data)
#Create figure object.

p = figure(title = 'Worldwide spread of Coronavirus', plot_height = 600 , plot_width = 1050)

p.xgrid.grid_line_color = None

p.ygrid.grid_line_color = None

#Add patch renderer to figure. 

patch=p.patches(xs='xs',ys='ys', source = geosource,fill_color = '#fff7bc',

          line_color = 'black', line_width = 0.35, fill_alpha = 1, 

                hover_fill_color="#fec44f")

p.add_tools(HoverTool(tooltips=[('Country','@country'),('ConfirmedCases','@confirmedcases'), ('Fatalities','@fatalities')], renderers=[patch]))



#Display figure inline in Jupyter Notebook.

output_notebook()

#Display figure.

show(p)
def getAlph(input):

  countries={}

  for country in pycountry.countries:

    countries[country.name] = country.alpha_3

    codes = countries.get(input, 'Unknown code')

  return codes
confirmed_cases_by_country['iso_alpha'] = confirmed_cases_by_country['Country'].apply(lambda x:getAlph(x))
confirmed_cases_by_country['TotalConfirmedCases'] = confirmed_cases_by_country['ConfirmedCases'].pow(0.3) * 3.5
confirmed_cases_by_country.head(10)
import plotly.express as px



fig = px.scatter_geo(confirmed_cases_by_country, locations="iso_alpha",color="Country",

                     text='Fatalities', size="TotalConfirmedCases",

                     projection="natural earth")

fig.update_layout(

    title={

        'text': "Hover on map to get deatails about Confirmed and Fatalities cases",

        'y':1,

        'x':0.4,

        'xanchor': 'center',

        'yanchor': 'top'})

fig.show()
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(x=worktrain.Date, y=worktrain['ConfirmedCases'], name="ConfirmedCases",

                         line_color='red'))



fig.add_trace(go.Scatter(x=worktrain.Date, y=worktrain['Fatalities'], name="Fatalities",

                        line_color='green'))



fig.update_layout(title_text='Covid-19 Cases Confimrd and Fatalities over time',xaxis_rangeslider_visible=True)

fig.show()
model_work = worktrain[['Country/Region','ConfirmedCases','Fatalities','Date']]
model_work['month'] = pd.DatetimeIndex(worktrain['Date']).month 

model_work['year'] = pd.DatetimeIndex(worktrain['Date']).year
model_work.head(10)
xtrain.isnull().sum()
model_work['PositiveCases'] = model_work['ConfirmedCases'].pow(0.3) * 3.5
model_work[['PositiveCases','Fatalities']].plot(figsize=(15,6))
temp_work = model_work[['PositiveCases','Fatalities']]
temp_work.index = model_work['Date']
temp_work[['PositiveCases']].plot(figsize=(15,4),color='red')
temp_work[['Fatalities']].plot(figsize=(15,4),color='green')
y1 = xtrain.iloc[:,-2].values

y2 = xtrain.iloc[:,-1].values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

xtrain = xtrain.apply(lambda col: le.fit_transform(col.astype(str)), axis=0, result_type='expand')

X = xtrain.iloc[:,1:5].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size = 0.20, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 4, metric = 'braycurtis', p = 1)

classifier.fit(X_train, y_train)
y_pred1 = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred1)

from sklearn.metrics import accuracy_score 

print( 'Accuracy Score confirmed cases :',accuracy_score(y_test,y_pred1)*100)
y1 = xtrain.iloc[:,-2].values

y2 = xtrain.iloc[:,-1].values
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

xtrain = xtrain.apply(lambda col: le.fit_transform(col.astype(str)), axis=0, result_type='expand')

X = xtrain.iloc[:,1:5].values
from sklearn.model_selection import train_test_split

X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y2, test_size = 0.25, random_state = 0)
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train2 = sc.fit_transform(X_train2)

X_test2 = sc.transform(X_test2)
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 4, metric = 'braycurtis', p = 1)

classifier.fit(X_train2, y_train2)
# Predicting the Test set results

y_pred2 = classifier.predict(X_test2)
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test2, y_pred2)

from sklearn.metrics import accuracy_score 

print( 'Accuracy Score fatality:',accuracy_score(y_test2,y_pred2)*100) 
xtrain = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv")

xtest = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/test.csv")
valid = xtrain[xtrain['Date'] >= xtest['Date'].min()]

xtrain = xtrain[xtrain['Date'] < xtest['Date'].min()]
from sklearn.linear_model import LinearRegression

from tqdm import tqdm_notebook as tqdm
log_target = True

plot = False



xtest['ConfirmedCases'] = np.nan

xtest['Fatalities'] = np.nan



countries = xtrain['Country/Region'].unique()

test_countries = xtest['Country/Region'].unique()



predictions = []

for c in tqdm(countries):

    xtrain_df = xtrain[xtrain['Country/Region'] == c]

    provinces = xtrain_df['Province/State'].unique()

    

    if c in test_countries:

        xtest_df = xtest[xtest['Country/Region'] == c]

        xtest_provinces = xtest_df['Province/State'].unique()

    

        for p in provinces:

            xtrain_df_p = xtrain_df[xtrain_df['Province/State'] == p]

            xtest_df_p = xtest_df[xtest_df['Province/State'] == p]

            

            confirmed = xtrain_df_p['ConfirmedCases'].values[-10:]

            fatalities = xtrain_df_p['Fatalities'].values[-10:]



            if log_target:

                confirmed = np.log1p(confirmed)

                fatalities = np.log1p(fatalities)



            if np.sum(confirmed) > 0:            

                x = np.arange(len(confirmed)).reshape(-1, 1)

                x_test = len(confirmed) + np.arange(len(xtest_df_p)).reshape(-1, 1)

                

                model = LinearRegression()

                model.fit(x, confirmed)

                p_conf = model.predict(x_test)

                p_conf = np.clip(p_conf, 0, None)

                p_conf = p_conf - np.min(p_conf) + confirmed[-1]

                if log_target:

                    p_conf = np.expm1(p_conf)

                xtest.loc[(xtest['Country/Region'] == c) & (xtest['Province/State'] == p), 'ConfirmedCases'] = p_conf

                

                model = LinearRegression()

                model.fit(x, fatalities)

                p_fatal = model.predict(x_test)

                p_fatal = np.clip(p_fatal, 0, None)

                p_fatal = p_fatal - np.min(p_fatal) + fatalities[-1]

                if log_target:

                    p_fatal = np.expm1(p_fatal)

                xtest.loc[(xtest['Country/Region'] == c) & (xtest['Province/State'] == p), 'Fatalities'] = p_fatal

                

                if plot:

                    plt.figure();

                    plt.plot(x, confirmed);

                    plt.plot(x, fatalities);

                    plt.plot(x_test, p_conf);

                    plt.plot(x_test, p_fatal);

                    plt.title(c + ', ' + p);

            

xtest[['ConfirmedCases', 'Fatalities']] = xtest[['ConfirmedCases', 'Fatalities']].fillna(0)
from sklearn.metrics import mean_squared_log_error
valid.sort_values(['Country/Region', 'Province/State', 'Date'], inplace=True)

preds = xtest.sort_values(['Country/Region', 'Province/State', 'Date'])

preds = valid[['Country/Region', 'Province/State', 'Date']].merge(preds, on=['Country/Region', 'Province/State', 'Date'], how='left')



score_c = np.sqrt(mean_squared_log_error(valid['ConfirmedCases'].values, preds['ConfirmedCases']))

score_f = np.sqrt(mean_squared_log_error(valid['Fatalities'].values, preds['Fatalities']))



print(f'score_c: {score_c}, score_f: {score_f}, mean: {np.mean([score_c, score_f])}')
pd.concat([valid.reset_index().drop('index', axis=1), 

           preds.reset_index()[['ConfirmedCases', 'Fatalities']].rename({'ConfirmedCases': 'ConfirmedCases_p', 'Fatalities': 'Fatalities_p'}, axis=1)], axis=1)
submission = xtest[['ForecastId', 'ConfirmedCases', 'Fatalities']]

submission.to_csv('submission.csv', index=False)

print(submission.shape)