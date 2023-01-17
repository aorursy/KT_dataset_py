import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
train = pd.read_csv('../input/covid19-global-forecasting-week-3/train.csv')

test = pd.read_csv('../input/covid19-global-forecasting-week-3/test.csv')
df = train
df['Province_State'] = df['Province_State'].fillna('Unkown State')

test['Province_State'] = test['Province_State'].fillna('Unkown State')
total_cases_datewise = df.groupby('Date')[['ConfirmedCases', 'Fatalities']].sum().reset_index()

total_cases_datewise = pd.melt(total_cases_datewise, id_vars=['Date'], value_vars=['ConfirmedCases', 'Fatalities'])

total_cases_datewise
# --- plotly ---

from plotly import tools, subplots

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

import plotly.figure_factory as ff

import plotly.io as pio

pio.templates.default = "plotly_dark"
fig = px.line(total_cases_datewise, x="Date", y="value", color="variable", title="worlwide cases trend")

fig.show()
lastDayCases = df[df['Date'] == '2020-04-04']

lastDayCases = lastDayCases[lastDayCases['ConfirmedCases']>5000]

countryWiseLastDayCases = lastDayCases.groupby(['Country_Region'])[['ConfirmedCases', 'Fatalities']].sum().reset_index()



countryWiseLastDayCases = countryWiseLastDayCases.sort_values('ConfirmedCases', ascending=False)



countryWiseLastDayCases = pd.melt(countryWiseLastDayCases, id_vars=['Country_Region'], value_vars=['ConfirmedCases', 'Fatalities'])



fig = px.bar(countryWiseLastDayCases.iloc[::-1],

             x='value', y='Country_Region', color='variable', barmode='group',

             title=f'Confirmed Cases/Deaths', text='value', height=1200, orientation='h')

fig.show()

countrywiseRiseInCases=df.groupby(['Date','Country_Region'])[['ConfirmedCases', 'Fatalities']].sum().reset_index()

fig = px.line(countrywiseRiseInCases, x="Date", y="ConfirmedCases", color="Country_Region", title="Contrywise Confirmed cases trend")

fig.show()
fig = px.line(countrywiseRiseInCases, x="Date", y="Fatalities", color="Country_Region", title="Contrywise Mortality trend")

fig.show()
test1 = df.loc[:,['Date', 'Country_Region', 'ConfirmedCases']].sort_values('Date')


fig = px.scatter_geo(test1, locations="Country_Region",locationmode ="country names", color="ConfirmedCases",

                     hover_name="Country_Region", size="ConfirmedCases",

                     animation_frame="Date",

                     projection="natural earth", title="Geospatial analysis of Confirmed cases")

fig.show()

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import matplotlib.animation as animation

from IPython.display import HTML

import string
test1 = df.groupby(['Date','Country_Region'])[['ConfirmedCases']].sum().reset_index()

test1['Country_Region'] = [s.replace('Taiwan*', 'Taiwan') for s in test1['Country_Region']]

test1['Country_Region'] = [s.replace('Korea, South', 'South Korea') for s in test1['Country_Region']]

test1['Country_Region'] = [s.replace('Congo (Brazzaville)', 'Brazzaville') for s in test1['Country_Region']]

test1['Country_Region'] = [s.replace('Congo (Kinshasa)', 'Kinshasa') for s in test1['Country_Region']]

test1['Country_Region'] = [s.replace('Congo (Kinshasa)', 'Kinshasa') for s in test1['Country_Region']]

minyear = test1['Date'].min()

dff = (test1[test1['Date'].eq(minyear)]

       .sort_values(by='ConfirmedCases', ascending=False))

       #.head(10))



fig, ax = plt.subplots(figsize=(15, 50))

dff=dff[::-1]

ax.barh(dff['Country_Region'], dff['ConfirmedCases'])

import seaborn as sns

palette = sns.color_palette(None, 180).as_hex()



colors = dict(zip(

    test1['Country_Region'].unique().tolist(),

    palette

))



colors
fig, ax = plt.subplots(figsize=(15, 8))

dff = dff[::-1]   # flip values from top to bottom

# pass colors values to `color=`

ax.barh(dff['Country_Region'], dff['ConfirmedCases'], color=[colors[x] for x in dff['Country_Region']])

# iterate over the values to plot labels and values (Tokyo, Asia, 38194.2)

for i, (name, value) in enumerate(zip(dff['Country_Region'], dff['ConfirmedCases'])):

    #print(i , value, name)

    ax.text(value+28, i,     name,            ha='right')  # Tokyo: name

    ax.text(value+7, i-0.4,     value,           ha='left')   # 38194.2: value

# Add year right middle portion of canvas

ax.text(1, 0.4, minyear, transform=ax.transAxes, size=16, ha='right')
minyear = test1['Date'].min()



fig, ax = plt.subplots(figsize=(15, 10))

def draw_barchart(date):

    dff = test1[test1['Date'].eq(date)].sort_values(by='ConfirmedCases', ascending=True).tail(20)

    ax.clear()

    ax.barh(dff['Country_Region'], dff['ConfirmedCases'], color=[colors[x] for x in dff['Country_Region']])

    dx = dff['ConfirmedCases'].max() / 200

    for i, (name, value) in enumerate(zip(dff['Country_Region'], dff['ConfirmedCases'])):

        ax.text(value-dx, i,     name,           size=14, weight=600, ha='right', va='bottom')

        ax.text(value+dx, i,     f'{value:,.0f}',  size=14, ha='left',  va='center')

        

    # ... polished styles

    ax.text(1, 0.4, date, transform=ax.transAxes, color='#777777', size=46, ha='right', weight=800)

    ax.text(0, 1.06, 'Confirmed cases (thousands)', transform=ax.transAxes, size=12, color='#777777')

    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    ax.xaxis.set_ticks_position('top')

    ax.tick_params(axis='x', colors='#777777', labelsize=12)

    ax.set_yticks([])

    ax.margins(0, 0.01)

    ax.grid(which='major', axis='x', linestyle='-')

    ax.set_axisbelow(True)

    ax.text(0, 1.12, 'The most Covid-19 cases countries in the world from Jan 2020 to April 2020',

            transform=ax.transAxes, size=24, weight=600, ha='left')

    ax.text(1, 0, 'by @Prem', transform=ax.transAxes, ha='right',

            color='#777777', bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))

    plt.box(False)

    

draw_barchart(minyear)
import matplotlib.animation as animation

from IPython.display import HTML

fig, ax = plt.subplots(figsize=(15, 10))

animator = animation.FuncAnimation(fig, draw_barchart, frames=test['Date'].unique().tolist())

HTML(animator.to_jshtml()) 

# oranimator.to_html5_video() or animator.save()
df['Country_Region'] = [s.replace('Taiwan*', 'Taiwan') for s in df['Country_Region']]

df['Country_Region'] = [s.replace('Korea, South', 'South Korea') for s in df['Country_Region']]

df['Country_Region'] = [s.replace('Congo (Brazzaville)', 'Brazzaville') for s in df['Country_Region']]

df['Country_Region'] = [s.replace('Congo (Kinshasa)', 'Kinshasa') for s in df['Country_Region']]

df['Country_Region'] = [s.replace('Congo (Kinshasa)', 'Kinshasa') for s in df['Country_Region']]



df1 = df.sort_values('Date')



df1_test = test.sort_values('Date')
df1_test
df1['year'] = pd.to_datetime(df1['Date']).dt.year

df1['month'] = pd.to_datetime(df1['Date']).dt.month

df1['day'] = pd.to_datetime(df1['Date']).dt.day



df1_test['year'] = pd.to_datetime(df1_test['Date']).dt.year

df1_test['month'] = pd.to_datetime(df1_test['Date']).dt.month

df1_test['day'] = pd.to_datetime(df1_test['Date']).dt.day
bridge_df = df1

bridge_df1 = df1_test

from xgboost import plot_importance, plot_tree

from sklearn.metrics import mean_squared_error, mean_absolute_error

from xgboost import XGBRegressor
bridge_df1
submission = []

df_train2 = bridge_df

X_train, y_train = df_train2.iloc[:,[6,7,8]], df_train2.iloc[:,4:6]

#model1 for predicting Confirmed Cases

model1 = XGBRegressor(n_estimators=1000)

model1.fit(X_train, y_train.iloc[:,0])

#model2 for predicting Fatalities

model2 = XGBRegressor(n_estimators=1000)

model2.fit(X_train, y_train.iloc[:,1])

#Get the test data for that particular country and state

df_test1 = bridge_df1

ForecastId = df_test1.ForecastId.values

#Remove the unwanted columns

df_test2 = df_test1.iloc[:,[4,5,6]]

#Get the predictions

y_pred1 = model1.predict(df_test2)

y_pred2 = model2.predict(df_test2)

#Append the predicted values to submission list

for i in range(len(y_pred1)):

    d = {'ForecastId':ForecastId[i], 'ConfirmedCases':y_pred1[i], 'Fatalities':y_pred2[i]}

    submission.append(d)
submissiondf = pd.DataFrame.from_dict(submission)

submissiondf.to_csv('submission.csv', index=False)
submissiondf