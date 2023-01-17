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
import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objs as go

import plotly.figure_factory as ff

import folium

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()   

# hide warnings

import warnings

warnings.filterwarnings('ignore')



pd.set_option('display.max_columns',500)

pd.set_option('display.max_rows',500)

cnf = '#393e46' # confirmed - grey

dth = '#ff2e63' # death - red

rec = '#21bf73' # recovered - cyan

act = '#fe9801' # active case - yellow
confirmed = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

recovered = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

death = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

covid19 = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

openLine = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv')

line = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv')
confirmed.head(2)
covid19.tail(2)
line[line['country']=='India']
covid19.info()
latestdate = str(covid19['ObservationDate'].max())
covid19['ObservationDate'] = pd.to_datetime(covid19['ObservationDate'])

covid19['Last Update'] = pd.to_datetime(covid19['Last Update'])
covid19['Active'] = covid19['Confirmed'] - (covid19['Recovered'] + covid19['Deaths'])

covid19.head()
### Lets check if there are any missing values

100*covid19.isna().sum()/covid19.shape[0]
today = covid19[covid19['ObservationDate'] == covid19['ObservationDate'].max()]

today.head()
len(covid19['Country/Region'].unique())
# Lets combine the data to country level

today_country_lvl = pd.DataFrame(today.groupby(by='Country/Region').sum()[['Active','Recovered','Deaths']])

today_country_lvl.head()
from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=2,subplot_titles=("WorldWide", "India"))

temp = pd.DataFrame(today_country_lvl.sum()).T

temp1 = today_country_lvl[today_country_lvl.index == 'India']

fig.add_trace(

    go.Bar(y=temp.get_values()[0], x=temp.columns, text = temp.get_values()[0],textposition='auto'

          ,marker=dict(color=[act,rec,dth])),

    row=1, col=1,

    

)



fig.add_trace(

    go.Bar(y=temp1.get_values()[0], x=temp1.columns, text = temp1.get_values()[0],textposition='auto',

          marker=dict(color=[act,rec,dth])),

    row=1, col=2

)



fig.update_layout(height=600, width=1000,title_text='Active vs Recovered vs Death as on '+latestdate)

fig.show()
cnf = '#393e46' # confirmed - grey

dth = '#ff2e63' # death - red

rec = '#21bf73' # recovered - cyan

act = '#fe9801' # active case - yellow

temp = today_country_lvl[today_country_lvl.index == 'India'].melt()

fig = px.treemap(temp, path=["variable"], values="value", height=600, width=800,

                 color_discrete_sequence=[act,rec,dth],title='India')

fig.show()
world_today = pd.DataFrame(today_country_lvl.sum()).T.melt()



fig = px.treemap(world_today, path=["variable"], values="value", height=600, width=800,

                 color_discrete_sequence=[act,rec,dth],title='WorldWide')

fig.show()
countryLevelData = pd.pivot_table(covid19, values=['Confirmed','Recovered','Deaths'], index=['ObservationDate','Country/Region'],

                     aggfunc=np.sum)

countryLevelData.reset_index(inplace=True)

countryLevelData['Week'] = countryLevelData['ObservationDate'].dt.week

countryLevelData['Month'] = countryLevelData['ObservationDate'].dt.month

countryLevelData.head()
countryWeekLevelData = pd.pivot_table(countryLevelData, values=['Confirmed','Recovered','Deaths','ObservationDate'], index=['Week','Country/Region'],

                     aggfunc=np.max).reset_index()

countryWeekLevelData[countryWeekLevelData['Country/Region']=='India']
china = countryWeekLevelData[countryWeekLevelData['Country/Region']=='Mainland China']

italy = countryWeekLevelData[countryWeekLevelData['Country/Region']=='Italy']

us = countryWeekLevelData[countryWeekLevelData['Country/Region']=='US']

india = countryWeekLevelData[countryWeekLevelData['Country/Region']=='India']





fig = make_subplots(rows=2, cols=2,subplot_titles=("China","Italy","US","India"))



fig.add_trace(

    go.Bar(

        y=china['Confirmed'], 

        x=china.Week - 3, 

        text = china['Confirmed'],

        textposition='auto'

    ),

    row=1, col=1,

    

)



fig.add_trace(

    go.Bar(

        y=italy['Confirmed'], 

        x=italy.Week - 4, 

        text = italy['Confirmed'],

        textposition='auto'

    ),

    row=1, col=2,    

)

fig.add_trace(

    go.Bar(

        y=us['Confirmed'], 

        x=us.Week - 3, 

        text = us['Confirmed'],

        textposition='auto'

    ),

    row=2, col=1,    

)

fig.add_trace(

    go.Bar(

        y=india['Confirmed'], 

        x=india.Week - 4, 

        text = india['Confirmed'],

        textposition='auto'

    ),

    row=2, col=2,    

)



fig.update_layout(

    title_text='Weekely Confirmed Cases from 2020-01-22 to '+latestdate,

    xaxis_title="Week",

    yaxis_title="Confirmed Cases",

)

fig.show()
china = countryWeekLevelData[countryWeekLevelData['Country/Region']=='Mainland China']

italy = countryWeekLevelData[countryWeekLevelData['Country/Region']=='Italy']

us = countryWeekLevelData[countryWeekLevelData['Country/Region']=='US']

india = countryWeekLevelData[countryWeekLevelData['Country/Region']=='India']





fig = make_subplots(rows=2, cols=2,subplot_titles=("China","Italy","US","India"))



fig.add_trace(

    go.Bar(

        y=china['Recovered'], 

        x=china.Week - 3, 

        text = china['Recovered'],

        textposition='auto'

    ),

    row=1, col=1,

    

)



fig.add_trace(

    go.Bar(

        y=italy['Recovered'], 

        x=italy.Week - 4, 

        text = italy['Recovered'],

        textposition='auto'

    ),

    row=1, col=2,    

)

fig.add_trace(

    go.Bar(

        y=us['Recovered'], 

        x=us.Week - 3, 

        text = us['Recovered'],

        textposition='auto'

    ),

    row=2, col=1,    

)

fig.add_trace(

    go.Bar(

        y=india['Recovered'], 

        x=india.Week - 4, 

        text = india['Recovered'],

        textposition='auto'

    ),

    row=2, col=2,    

)



fig.update_layout(

    title_text='Weekely Recovered Cases from 2020-01-22 to '+latestdate,

    xaxis_title="Week",

    yaxis_title="Recovered Cases",

)

fig.show()
china = countryWeekLevelData[countryWeekLevelData['Country/Region']=='Mainland China']

italy = countryWeekLevelData[countryWeekLevelData['Country/Region']=='Italy']

us = countryWeekLevelData[countryWeekLevelData['Country/Region']=='US']

india = countryWeekLevelData[countryWeekLevelData['Country/Region']=='India']





fig = make_subplots(rows=2, cols=2,subplot_titles=("China","Italy","US","India"))



fig.add_trace(

    go.Bar(

        y=china['Deaths'], 

        x=china.Week - 3, 

        text = china['Deaths'],

        textposition='auto'

    ),

    row=1, col=1,

    

)



fig.add_trace(

    go.Bar(

        y=italy['Deaths'], 

        x=italy.Week - 4, 

        text = italy['Deaths'],

        textposition='auto'

    ),

    row=1, col=2,    

)

fig.add_trace(

    go.Bar(

        y=us['Deaths'], 

        x=us.Week - 3, 

        text = us['Deaths'],

        textposition='auto'

    ),

    row=2, col=1,    

)

fig.add_trace(

    go.Bar(

        y=india['Deaths'], 

        x=india.Week - 4, 

        text = india['Deaths'],

        textposition='auto'

    ),

    row=2, col=2,    

)



fig.update_layout(

    title_text='Weekely Deaths Cases from 2020-01-22 to '+latestdate,

    xaxis_title="Week",

    yaxis_title="Deaths Cases",

)

fig.show()
fig = go.Figure(data=[

    go.Bar(name='Death', x=india['Week']-4, y=india['Deaths'],text=india['Deaths'],textposition='auto'),

    go.Bar(name='Recovered', x=india['Week']-4, y=india['Recovered'], text=india['Recovered'],textposition='auto'),

    go.Bar(name='Active', x=india['Week']-4, y=india['Confirmed']-india['Recovered']-india['Deaths'],text = india['Confirmed']-india['Recovered']-india['Deaths'],

        textposition='auto')

])

# Change the bar mode

fig.update_layout(barmode='stack',

                title_text='Weekely Corona Cases in India from 2020-01-22 to '+latestdate,

                xaxis_title="Week",

                yaxis_title="Cases",height=600, width=900,

                 )

fig.show()



fig = go.Figure(data=[

    go.Bar(name='Death', x=china['Week']-4, y=china['Deaths'],text=china['Deaths'],textposition='auto'),

    go.Bar(name='Recovered', x=china['Week']-4, y=china['Recovered'],text=china['Recovered'],textposition='auto'),

    go.Bar(name='Active', x=china['Week']-4, y=china['Confirmed']-china['Recovered']-china['Deaths'],text = china['Confirmed']-china['Recovered']-china['Deaths'],

        textposition='auto')

])

# Change the bar mode

fig.update_layout(barmode='stack',

                title_text='Weekely Corona Cases in China from 2020-01-22 to '+latestdate,

                xaxis_title="Week",

                yaxis_title="Cases",height=600, width=900,

                 )

fig.show()
import matplotlib.ticker as ticker

import matplotlib.animation as animation

from IPython.display import HTML
countryLevelData['ObservationDate'] = countryLevelData['ObservationDate'].map(lambda x: str(x))


fig = px.bar(countryLevelData[countryLevelData['Country/Region']!='Mainland China'], y="Country/Region", x="Confirmed", color="Country/Region",

  animation_frame="ObservationDate", animation_group="Country/Region",orientation='h',text='Confirmed',title='Confirmed Cases Worldwide')

fig.show()


fig = px.bar(countryLevelData[countryLevelData['Country/Region']!='Mainland China'], y="Country/Region", x="Recovered", color="Country/Region",

  animation_frame="ObservationDate", animation_group="Country/Region",orientation='h',text='Recovered',title='Recovered Cases Worldwide')

fig.show()
WW = covid19.groupby(by='ObservationDate').sum()[['Confirmed','Deaths','Recovered']]
WW = WW.reset_index()

WW.head()
WW['Day'] = range(1,WW.shape[0]+1)
X = WW[['Day']]

y = WW['Confirmed']

from sklearn.model_selection import train_test_split

X_train = X[0 : int(0.8*WW.shape[0])]

y_train = y[0 : int(0.8*WW.shape[0])]



X_test = X[int(0.8*WW.shape[0]) : ]

y_test = y[int(0.8*WW.shape[0]) : ]
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train,y_train)
res_lr = X_test.copy()

res_lr['Actual'] = y_test

res_lr['Predicted'] = lr.predict(X_test)

res_lr.head()
from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline

model = Pipeline([('poly', PolynomialFeatures(degree=4)),

                  ('linear', LinearRegression(fit_intercept=False))])

# fit to an order-3 polynomial data

x = np.arange(5)

y = 3 - 2 * x + x ** 2 - x ** 3

model = model.fit(X_train,y_train)

model.named_steps['linear'].coef_
model.predict(X_test)
res_poly = X_test.copy()

res_poly['Actual'] = y_test

res_poly['Predicted'] = model.predict(X_test)

res_poly.head()