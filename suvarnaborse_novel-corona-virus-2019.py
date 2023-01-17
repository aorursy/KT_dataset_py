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



# Any results you write to the current directory are saved as output.
import plotly.graph_objects as go

import plotly.express as px

import plotly.io as pio

pio.templates.default = "plotly_dark"

from plotly.subplots import make_subplots

import seaborn as sns



from sklearn import preprocessing

import time

from datetime import datetime

import warnings

warnings.filterwarnings('ignore')



import lightgbm as lgb

import xgboost as xgb

from xgboost import plot_importance, plot_tree

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn import linear_model

from sklearn.metrics import mean_squared_error

from sklearn import metrics

from sklearn.model_selection import train_test_split
sub = pd.read_csv(r'/kaggle/input/covid19-global-forecasting-week-1/submission.csv')

sub.to_csv('submission.csv', index = False)
test = pd.read_csv(r'/kaggle/input/covid19-global-forecasting-week-1/test.csv')

test.tail()
train = pd.read_csv(r'/kaggle/input/covid19-global-forecasting-week-1/train.csv', parse_dates = ['Date'])

train.head()
df = pd.read_csv(r'/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv', parse_dates = ['ObservationDate'])

df.head()
test.rename(columns = {'Province/State':'State','Country/Region':'Country'}, inplace = True)

train.rename(columns = {'Province/State':'State','Country/Region':'Country','ConfirmedCases':'Confirmed',

                        'Fatalities':'Deaths'}, inplace = True)

df.rename(columns = {'ObservationDate':'Date','Province/State':'State','Country/Region':'Country'}, inplace = True)
print(df.shape)

print(train.shape)

print(test.shape)
cases = ['Confirmed', 'Deaths', 'Recovered', 'Active']



df['Active'] = df['Confirmed'] - df['Deaths'] - df['Recovered']

df['Country'] = df['Country'].replace('Mainland China', 'China')

df[['State']] = df[['State']].fillna('')

df[cases] = df[cases].fillna(0)
confirmiedcases = pd.DataFrame(df.groupby('Country')['Confirmed'].sum())

confirmiedcases['Country'] = confirmiedcases.index

confirmiedcases.index = np.arange(1,204)



Deathcases = pd.DataFrame(df.groupby('Country')['Deaths'].sum())

Deathcases['Country'] = Deathcases.index

Deathcases.iodex = np.arange(1,184)



Recoveredcases = pd.DataFrame(df.groupby('Country')['Recovered'].sum())

Recoveredcases['Country'] = Recoveredcases.index

Recoveredcases.iodex = np.arange(1,204)



Activecases = pd.DataFrame(df.groupby('Country')['Active'].sum())

Activecases['Country'] = Activecases.index

Activecases.iodex = np.arange(1,184)



global_Activecases = Activecases[['Country','Active']]

global_Deathcases = Deathcases[['Country','Deaths']]

global_Recoveredcases = Recoveredcases[['Country','Recovered']]

global_confirmiedcases = confirmiedcases[['Country','Confirmed']]



fig = px.bar(global_confirmiedcases.sort_values('Confirmed',ascending=False)[:20][::-1],x='Confirmed',y='Country',title='Confirmed Cases Worldwide',text='Confirmed', height=900, orientation='h')

fig.show()



fig = px.bar(global_Deathcases.sort_values('Deaths',ascending=False)[:20][::-1],x='Deaths',y='Country',title='Deaths Cases Worldwide',text='Deaths', height=900, orientation='h')

fig.show()



fig = px.bar(global_Recoveredcases.sort_values('Recovered',ascending=False)[:20][::-1],x='Recovered',y='Country',title='Recovered Cases Worldwide',text='Recovered', height=900, orientation='h')

fig.show()



fig = px.bar(global_Activecases.sort_values('Active',ascending=False)[:20][::-1],x='Active',y='Country',title='Active Cases Worldwide',text='Active', height=900, orientation='h')

fig.show()
date_c = df.groupby('Date')['Confirmed','Deaths','Recovered','Active'].sum().reset_index()





from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=4, subplot_titles=("Comfirmed", "Deaths", "Recovered",'Active'))



trace1 = go.Scatter(

                x=date_c['Date'],

                y=date_c['Confirmed'],

                name="Confirmed",

                line_color='orange',

                mode='lines+markers',

                opacity=0.8)

trace2 = go.Scatter(

                x=date_c['Date'],

                y=date_c['Deaths'],

                name="Deaths",

                line_color='red',

                mode='lines+markers',

                opacity=0.8)



trace3 = go.Scatter(

                x=date_c['Date'],

                y=date_c['Recovered'],

                name="Recovered",

                mode='lines+markers',

                line_color='green',

                opacity=0.8)



trace4 = go.Scatter(

                x=date_c['Date'],

                y=date_c['Active'],

                name="Active",

                line_color='blue',

                mode='lines+markers',

                opacity=0.8)



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)

fig.append_trace(trace3, 1, 3)

fig.append_trace(trace4, 1, 4)

fig.update_layout(template="plotly_dark",title_text = '<b>Global Spread of the Coronavirus Over Time </b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='white'))

fig.show()

train_dataset = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

drop_clo = ['Province/State','Country/Region','Lat','Long']

train_dataset=train_dataset.drop(drop_clo,axis=1)

datewise= list(train_dataset.columns)

val_dataset = train_dataset[datewise[-30:]]
fig = make_subplots(rows=3, cols=1)



fig.add_trace(

    go.Scatter(x=np.arange(70), mode='lines+markers', y=train_dataset.loc[0].values, 

               marker=dict(color="dodgerblue"), showlegend=False,),

                row=1, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70, 100), y=val_dataset.loc[0].values, mode='lines+markers',

               marker=dict(color="darkorange"), showlegend=False,),

               row=1, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70), mode='lines+markers', y=train_dataset.loc[1].values,

               marker=dict(color="dodgerblue"), showlegend=False),

               row=2, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70, 100), y=val_dataset.loc[1].values, mode='lines+markers', 

               marker=dict(color="darkorange"), showlegend=False),

    row=2, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70), mode='lines+markers', y=train_dataset.loc[2].values, 

               marker=dict(color="dodgerblue"), showlegend=False),

    row=3, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70, 100), y=val_dataset.loc[2].values, mode='lines+markers', 

               marker=dict(color="darkorange"), showlegend=False),

    row=3, col=1

)



fig.update_layout(height=1200, width=800, title_text="Train (blue) vs. Validation (orange) sales")

fig.show()
predictions = []

for i in range(len(val_dataset.columns)):

    if i == 0:

        predictions.append(train_dataset[train_dataset.columns[-1]].values)

    else:

        predictions.append(val_dataset[val_dataset.columns[i-1]].values)

    

predictions = np.transpose(np.array([row.tolist() for row in predictions]))

error_naive = np.linalg.norm(predictions[:3] - val_dataset.values[:3])/len(predictions[0])
#ConfirmedCases = predictions

ConfirmedCases = predictions.flatten()

ConfirmedCases = ConfirmedCases.astype(int)

ConfirmedCases
pred_1 = predictions[0]

pred_2 = predictions[1]

pred_3 = predictions[2]



fig = make_subplots(rows=3, cols=1)



fig.add_trace(

    go.Scatter(x=np.arange(70), mode='lines', y=train_dataset.loc[0].values, 

               marker=dict(color="dodgerblue"),

               name="Train"),

    row=1, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70, 100), y=val_dataset.loc[0].values, mode='lines', 

               marker=dict(color="darkorange"),

               name="Val"),

    row=1, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70, 100), y=pred_1, mode='lines',

               marker=dict(color="seagreen"),

               name="Pred"),

    row=1, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70), mode='lines', y=train_dataset.loc[1].values, 

               marker=dict(color="dodgerblue"), showlegend=False),

    row=2, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70, 100), y=val_dataset.loc[1].values, mode='lines', 

               marker=dict(color="darkorange"), showlegend=False),

    row=2, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70, 100), y=pred_2, mode='lines', marker=dict(color="seagreen"), 

               showlegend=False,

               name="Denoised signal"),

    row=2, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70), mode='lines', y=train_dataset.loc[2].values, 

               marker=dict(color="dodgerblue"), showlegend=False),

    row=3, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70, 100), y=val_dataset.loc[2].values, mode='lines', 

               marker=dict(color="darkorange"), showlegend=False),

    row=3, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70, 100), y=pred_3, mode='lines', marker=dict(color="seagreen"), 

               showlegend=False,

               name="Denoised signal"),

    row=3, col=1

)



fig.update_layout(height=1200, width=800, title_text="Naive approach")

fig.show()
predictions = []

for i in range(len(val_dataset.columns)):

    if i == 0:

        predictions.append(np.mean(train_dataset[train_dataset.columns[-30:]].values, axis=1))

    if i < 31 and i > 0:

        predictions.append(0.5 * (np.mean(train_dataset[train_dataset.columns[-30+i:]].values, axis=1) + \

                                  np.mean(predictions[:i], axis=0)))

    if i > 31:

        predictions.append(np.mean([predictions[:i]], axis=1))

    

predictions = np.transpose(np.array([row.tolist() for row in predictions]))

error_avg = np.linalg.norm(predictions[:3] - val_dataset.values[:3])/len(predictions[0])
pred_1 = predictions[0]

pred_2 = predictions[1]

pred_3 = predictions[2]



fig = make_subplots(rows=3, cols=1)



fig.add_trace(

    go.Scatter(x=np.arange(70), mode='lines', y=train_dataset.loc[0].values,

               marker=dict(color="dodgerblue"),

               name="Train"),

    row=1, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70, 100), y=val_dataset.loc[0].values, mode='lines', 

               marker=dict(color="darkorange"),

               name="Val"),

    row=1, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70, 100), y=pred_1, mode='lines', marker=dict(color="seagreen"),

               name="Pred"),

    row=1, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70), mode='lines', y=train_dataset.loc[1].values,

               marker=dict(color="dodgerblue"), showlegend=False),

    row=2, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70, 100), y=val_dataset.loc[1].values, mode='lines', 

               marker=dict(color="darkorange"), showlegend=False),

    row=2, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70, 100), y=pred_2, mode='lines', marker=dict(color="seagreen"),

               showlegend=False,

               name="Denoised signal"),

    row=2, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70), mode='lines', y=train_dataset.loc[2].values, 

               marker=dict(color="dodgerblue"), showlegend=False),

    row=3, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70, 100), y=val_dataset.loc[2].values, mode='lines', 

               marker=dict(color="darkorange"), showlegend=False),

    row=3, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70, 100), y=pred_3, mode='lines', marker=dict(color="seagreen"), 

               showlegend=False,

               name="Denoised signal"),

    row=3, col=1

)



fig.update_layout(height=1200, width=800, title_text="Moving average")

fig.show()
train_dataset = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

drop_clo = ['Province/State','Country/Region','Lat','Long']

train_dataset=train_dataset.drop(drop_clo,axis=1)

datewise= list(train_dataset.columns)

val_dataset = train_dataset[datewise[-30:]]
fig = make_subplots(rows=3, cols=1)



fig.add_trace(

    go.Scatter(x=np.arange(70), mode='lines+markers', y=train_dataset.loc[0].values, 

               marker=dict(color="dodgerblue"), showlegend=False,),

                row=1, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70, 100), y=val_dataset.loc[0].values, mode='lines+markers',

               marker=dict(color="darkorange"), showlegend=False,),

               row=1, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70), mode='lines+markers', y=train_dataset.loc[1].values,

               marker=dict(color="dodgerblue"), showlegend=False),

               row=2, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70, 100), y=val_dataset.loc[1].values, mode='lines+markers', 

               marker=dict(color="darkorange"), showlegend=False),

    row=2, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70), mode='lines+markers', y=train_dataset.loc[2].values, 

               marker=dict(color="dodgerblue"), showlegend=False),

    row=3, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70, 100), y=val_dataset.loc[2].values, mode='lines+markers', 

               marker=dict(color="darkorange"), showlegend=False),

    row=3, col=1

)



fig.update_layout(height=1200, width=800, title_text="Train (blue) vs. Validation (orange) sales")

fig.show()
predictions = []

for i in range(len(val_dataset.columns)):

    if i == 0:

        predictions.append(train_dataset[train_dataset.columns[-1]].values)

    else:

        predictions.append(val_dataset[val_dataset.columns[i-1]].values)

    

predictions = np.transpose(np.array([row.tolist() for row in predictions]))

error_naive = np.linalg.norm(predictions[:3] - val_dataset.values[:3])/len(predictions[0])
Fatalities = predictions.flatten()

Fatalities.shape

Fatalities = Fatalities.astype(int)

Fatalities
pred_1 = predictions[0]

pred_2 = predictions[1]

pred_3 = predictions[2]



fig = make_subplots(rows=3, cols=1)



fig.add_trace(

    go.Scatter(x=np.arange(70), mode='lines', y=train_dataset.loc[0].values, 

               marker=dict(color="dodgerblue"),

               name="Train"),

    row=1, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70, 100), y=val_dataset.loc[0].values, mode='lines', 

               marker=dict(color="darkorange"),

               name="Val"),

    row=1, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70, 100), y=pred_1, mode='lines',

               marker=dict(color="seagreen"),

               name="Pred"),

    row=1, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70), mode='lines', y=train_dataset.loc[1].values, 

               marker=dict(color="dodgerblue"), showlegend=False),

    row=2, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70, 100), y=val_dataset.loc[1].values, mode='lines', 

               marker=dict(color="darkorange"), showlegend=False),

    row=2, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70, 100), y=pred_2, mode='lines', marker=dict(color="seagreen"), 

               showlegend=False,

               name="Denoised signal"),

    row=2, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70), mode='lines', y=train_dataset.loc[2].values, 

               marker=dict(color="dodgerblue"), showlegend=False),

    row=3, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70, 100), y=val_dataset.loc[2].values, mode='lines', 

               marker=dict(color="darkorange"), showlegend=False),

    row=3, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70, 100), y=pred_3, mode='lines', marker=dict(color="seagreen"), 

               showlegend=False,

               name="Denoised signal"),

    row=3, col=1

)



fig.update_layout(height=1200, width=800, title_text="Naive approach")

fig.show()
predictions = []

for i in range(len(val_dataset.columns)):

    if i == 0:

        predictions.append(np.mean(train_dataset[train_dataset.columns[-30:]].values, axis=1))

    if i < 31 and i > 0:

        predictions.append(0.5 * (np.mean(train_dataset[train_dataset.columns[-30+i:]].values, axis=1) + \

                                  np.mean(predictions[:i], axis=0)))

    if i > 31:

        predictions.append(np.mean([predictions[:i]], axis=1))

    

predictions = np.transpose(np.array([row.tolist() for row in predictions]))

error_avg = np.linalg.norm(predictions[:3] - val_dataset.values[:3])/len(predictions[0])
pred_1 = predictions[0]

pred_2 = predictions[1]

pred_3 = predictions[2]



fig = make_subplots(rows=3, cols=1)



fig.add_trace(

    go.Scatter(x=np.arange(70), mode='lines', y=train_dataset.loc[0].values,

               marker=dict(color="dodgerblue"),

               name="Train"),

    row=1, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70, 100), y=val_dataset.loc[0].values, mode='lines', 

               marker=dict(color="darkorange"),

               name="Val"),

    row=1, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70, 100), y=pred_1, mode='lines', marker=dict(color="seagreen"),

               name="Pred"),

    row=1, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70), mode='lines', y=train_dataset.loc[1].values,

               marker=dict(color="dodgerblue"), showlegend=False),

    row=2, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70, 100), y=val_dataset.loc[1].values, mode='lines', 

               marker=dict(color="darkorange"), showlegend=False),

    row=2, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70, 100), y=pred_2, mode='lines', marker=dict(color="seagreen"),

               showlegend=False,

               name="Denoised signal"),

    row=2, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70), mode='lines', y=train_dataset.loc[2].values, 

               marker=dict(color="dodgerblue"), showlegend=False),

    row=3, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70, 100), y=val_dataset.loc[2].values, mode='lines', 

               marker=dict(color="darkorange"), showlegend=False),

    row=3, col=1

)



fig.add_trace(

    go.Scatter(x=np.arange(70, 100), y=pred_3, mode='lines', marker=dict(color="seagreen"), 

               showlegend=False,

               name="Denoised signal"),

    row=3, col=1

)



fig.update_layout(height=1200, width=800, title_text="Moving average")

fig.show()
a = np.arange(1,12213,1)

ForecastId = a

ForecastId
ConfirmedCases = ConfirmedCases[0:12212]

Fatalities = Fatalities[0:12212]
d = {'ForecastId':ForecastId,'ConfirmedCases':ConfirmedCases,'Fatalities':Fatalities}

dff = pd.DataFrame(data=d)

dff.head()
dff.to_csv('submission.csv', index=False)
test.head()
print("External Data")

print(f"Earliest Entry: {train['Date'].min()}")

print(f"Last Entry:     {train['Date'].max()}")

print(f"Total Days:     {train['Date'].max() - train['Date'].min()}")
grp = train.groupby('Date')['Date','Confirmed','Deaths'].sum().reset_index()



fig = px.line(grp, x = 'Date', y = 'Confirmed',title = 'Worldwide Confirmed Cases')

fig.show()



fig = px.line(grp, x = 'Date',y = 'Confirmed', title = 'WorldWide Confirmed Cases on logarithmic cases',

              log_y = True)

fig.show()
grp_china = train[train['Country'] == 'China']

grp_china_date = grp_china.groupby('Date')['Date','Confirmed','Deaths'].sum().reset_index()



grp_italy = train[train['Country'] == 'Italy']

grp_italy_date = grp_italy.groupby('Date')['Date','Confirmed','Deaths'].sum().reset_index()



grp_us = train[train['Country'] == 'US']

grp_us_date = grp_us.groupby('Date')['Date','Confirmed','Deaths'].sum().reset_index()



grp_india = train[train['Country'] == 'India']

grp_india_date = grp_india.groupby('Date')['Date','Confirmed','Deaths'].sum().reset_index()



grp_rest = train[~train['Country'].isin(['China','Italy','US','India'])].reset_index()

grp_rest_date = grp_rest.groupby('Date')['Date','Confirmed','Deaths'].sum().reset_index()
plot_titles = ['China', 'Italy', 'USA','India' , 'Rest of the World']



fig = px.line(grp_china_date, x="Date", y="Confirmed", 

              title=f"Confirmed Cases in {plot_titles[0].upper()} Over Time", 

              color_discrete_sequence=['blue'],

              height=500

             )

fig.show()



fig = px.line(grp_italy_date, x="Date", y="Confirmed", 

              title=f"Confirmed Cases in {plot_titles[1].upper()} Over Time", 

              color_discrete_sequence=['pink'],

              height=500

             )

fig.show()



fig = px.line(grp_us_date, x="Date", y="Confirmed", 

              title=f"Confirmed Cases in {plot_titles[2].upper()} Over Time", 

              color_discrete_sequence=['brown'],

              height=500

             )

fig.show()



fig = px.line(grp_india, x = 'Date', y = 'Confirmed',

             title = f"Confirmed Cases in {plot_titles[3].upper()} Over Time",

             color_discrete_sequence = ['orange'],

             height=500

             )

fig.show()



fig = px.line(grp_rest_date, x="Date", y="Confirmed", 

              title=f"Confirmed Cases in {plot_titles[4].upper()} Over Time", 

              color_discrete_sequence=['red'],

              height=500

             )

fig.show()

train['State'] = train['State'].fillna('')

temp = train[[col for col in train.columns if col != 'State']]



latest = temp[temp['Date'] == max(temp['Date'])].reset_index()

latest_grp = latest.groupby('Country')['Confirmed', 'Deaths'].sum().reset_index()
fig = px.choropleth(latest_grp, locations="Country", 

                    locationmode='country names', color="Confirmed", 

                    hover_name="Country", range_color=[1,5000], 

                    color_continuous_scale="peach", 

                    title='Countries with Confirmed Cases')



fig.show()
europe = list(['Austria','Belgium','Bulgaria','Croatia','Cyprus','Czechia','Denmark','Estonia','Finland','France','Germany','Greece','Hungary','Ireland',

               'Italy', 'Latvia','Luxembourg','Lithuania','Malta','Norway','Netherlands','Poland','Portugal','Romania','Slovakia','Slovenia',

               'Spain', 'Sweden', 'United Kingdom', 'Iceland', 'Russia', 'Switzerland', 'Serbia', 'Ukraine', 'Belarus',

               'Albania', 'Bosnia and Herzegovina', 'Kosovo', 'Moldova', 'Montenegro', 'North Macedonia'])



europe_grp_latest = latest_grp[latest_grp['Country'].isin(europe)]
europe_grp_latest.head()
fig = px.choropleth(europe_grp_latest, locations="Country", 

                    locationmode='country names', color="Confirmed", 

                    hover_name="Country", range_color=[1,2000], 

                    color_continuous_scale='portland', 

                    title='European Countries with Confirmed Cases', scope='europe', height=800)

#fig.update(layout_coloraxis_showscale=False)

fig.show()
fig = px.bar(latest_grp.sort_values('Confirmed', ascending=False)[:20][::-1], 

             x='Confirmed', y='Country',

             title='Confirmed Cases Worldwide', text='Confirmed', height=1000, orientation='h')

fig.show()
fig = px.bar(europe_grp_latest.sort_values('Confirmed', ascending=False)[:10][::-1], 

             x='Confirmed', y='Country', color_discrete_sequence=['#84DCC6'],

             title='Confirmed Cases in Europe', text='Confirmed', orientation='h')

fig.show()
usa = df[df['Country'] == "US"]

usa_latest = usa[usa['Date'] == max(usa['Date'])]

usa_latest = usa_latest.groupby('State')['Confirmed', 'Deaths'].max().reset_index()



fig = px.bar(usa_latest.sort_values('Confirmed', ascending=False)[:10][::-1], 

             x='Confirmed', y='State', color_discrete_sequence=['#D63230'],

             title='Confirmed Cases in USA', text='Confirmed', orientation='h')

fig.show()
fig = px.line(grp, x="Date", y="Deaths", title="Worldwide Deaths Over Time",

             color_discrete_sequence=['#F42272'])

fig.show()



fig = px.line(grp, x="Date", y="Deaths", title="Worldwide Deaths (Logarithmic Scale) Over Time", 

              log_y=True, color_discrete_sequence=['#F42272'])

fig.show()
plot_titles = ['China', 'Italy', 'USA','India','Rest of the World']



fig = px.line(grp_china_date, x="Date", y="Deaths", 

              title=f"Deaths in {plot_titles[0].upper()} Over Time", 

              color_discrete_sequence=['#F61067'],

              height=500

             )

fig.show()



fig = px.line(grp_italy_date, x="Date", y="Deaths", 

              title=f"Deaths in {plot_titles[1].upper()} Over Time", 

              color_discrete_sequence=['#91C4F2'],

              height=500

             )

fig.show()



fig = px.line(grp_us_date, x="Date", y="Deaths", 

              title=f"Deaths in {plot_titles[2].upper()} Over Time", 

              color_discrete_sequence=['#6F2DBD'],

              height=500

             )

fig.show()



fig = px.line(grp_india_date, x = 'Date', y = 'Deaths',

             title = f"Deaths in {plot_titles[3].upper()} Over Time",

             color_discrete_sequence=['pink'],

             height=500

             )

fig.show()



fig = px.line(grp_rest_date, x="Date", y="Deaths", 

              title=f"Deaths in {plot_titles[4].upper()} Over Time", 

              color_discrete_sequence=['#FFDF64'],

              height=500

             )

fig.show()
fig = px.choropleth(latest_grp, locations="Country", 

                    locationmode='country names', color="Deaths", 

                    hover_name="Deaths", range_color=[1,100], 

                    color_continuous_scale="peach", 

                    title='Countries with Reported Deaths')

# fig.update(layout_coloraxis_showscale=False)

fig.show()

fig = px.choropleth(europe_grp_latest, locations="Country", 

                    locationmode='country names', color="Deaths", 

                    hover_name="Country", range_color=[1,100], 

                    color_continuous_scale='portland',

                    title='Reported Deaths in EUROPE', scope='europe', height=800)



fig.show()
fig = px.bar(latest_grp.sort_values('Deaths', ascending=False)[:10][::-1], 

             x='Deaths', y='Country',

             title='Confirmed Deaths Worldwide', text='Deaths', orientation='h')

fig.show()