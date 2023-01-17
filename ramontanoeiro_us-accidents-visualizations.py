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
cols = ['Severity', 'Start_Time', 'City', 'County', 'State', 'Temperature(F)',

        'Humidity(%)', 'Precipitation(in)', 'Weather_Condition',

       'Traffic_Signal', 'Sunrise_Sunset']

## Remember to tround latitude and longitude to 2 digits, we're interested in checking if any specific place may has more accidents than others.
df = pd.read_csv("../input/us-accidents/US_Accidents_June20.csv", usecols=cols)
df.head()
df.info()
df.isnull().sum()
df['Precipitation(in)'].fillna(0, inplace=True)
df.dropna(axis=0, inplace=True)
df['Start_Time'] = pd.to_datetime(df['Start_Time'])
## Extracting each data from the date. Separating it in it's own column will help with visualizations.

df['Week_day'] = df['Start_Time'].dt.dayofweek

df['Month'] = df['Start_Time'].dt.month

df['Year'] = df['Start_Time'].dt.year
df['Week_day'] = df['Week_day'].map({0:'Monday', 1:'Tuesday', 2: 'Wednesday', 3:'Thrusday', 4:'Friday', 5:'Saturday', 6:'Sunday'})

df['Month'] = df['Month'].map({1:'January', 2:'February', 3: 'March', 4:'April', 5:'May', 6:'June', 7:'July',

                                  8:'August', 9:'September', 10: 'October', 11:'November', 12:'December'})
df.drop("Start_Time", axis=1, inplace=True)
df.head()
df.head()
import plotly

import plotly.graph_objs as go

import plotly.offline as py
plotly.offline.init_notebook_mode(connected=True)
severity = df['Severity'].value_counts()

severity
#Selecting data to plot

data = [go.Pie(labels=severity.index,

               values=severity.values,

               direction='clockwise')

       ]



# Editing style

layout = go.Layout(title='Severity of accidents',

                   width=600,

                   height=600                   

                  )

#Creating figure

fig = go.Figure(data=data, layout=layout)



## Plotting

py.iplot(fig)
years = df['Year'].groupby(df['Year']).count()

accidents_years = pd.DataFrame(years)

accidents_years
data = [go.Bar(x=[2016,2017,2018,2019,2020],

               y=accidents_years['Year'])]



layout = go.Layout(title='Accidents by year 2016-June 2020',

                   xaxis={'title':'Year'},

                   yaxis={'title':'Number of accidents'},

                   width=700,

                   height=600)





fig = go.Figure(data=data, layout=layout)

fig.update_yaxes(nticks=4)



py.iplot(fig)
m2016 = df['Month'].groupby(by=df['Month'].loc[df['Year']==2016]).count()

accidents_months_2016 = pd.DataFrame(m2016)



m2017 = df['Month'].groupby(by=df['Month'].loc[df['Year']==2017]).count()

accidents_months_2017 = pd.DataFrame(m2017)



m2018 = df['Month'].groupby(df['Month'].loc[df['Year']==2018]).count()

accidents_months_2018 = pd.DataFrame(m2018)



m2019 = df['Month'].groupby(df['Month'].loc[df['Year']==2019]).count()

accidents_months_2019 = pd.DataFrame(m2019)



m2020 = df['Month'].groupby(df['Month'].loc[df['Year']==2020]).count()

accidents_months_2020 = pd.DataFrame(m2020)
accidents_months_2016
months_2016 = {'Month of the year - 2016': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],

               'Total Accidents':          [0,          973,        6150,    17636,   17141, 29603,  44490,  54813,    53074,       53774,     62890,      57256]}

accidents_months_2016 = pd.DataFrame(months_2016)

accidents_months_2016
accidents_months_2017
months_2017 = {'Month of the year - 2017': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],

               'Total Accidents':          [53331,      49607,      55660,   46857,   40166, 44971,  42048,  78296,    73586,       72633,     67894,      69380]}

accidents_months_2017 = pd.DataFrame(months_2017)

accidents_months_2017
accidents_months_2018
months_2018 = {'Month of the year - 2018': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],

              'Total Accidents':           [72419,      69290,      72444,   70988,   74459, 62367,   64017,  74482,    71202,       84941,     80120,      68245]}

accidents_months_2018 = pd.DataFrame(months_2018)

accidents_months_2018
accidents_months_2019
months_2019 = {'Month of the year - 2019': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],

               'Total Accidents':          [77149,      72674,      67672,   71887,   72524, 64183,  64925,  73417,    85334,       103757,    79952,      95450]}

accidents_months_2019 = pd.DataFrame(months_2019)

accidents_months_2019
accidents_months_2020
months_2020 = {'Month of the year - 2020': ['January', 'February', 'March', 'April', 'May', 'June'],

               'Total Accidents':          [89506,      83634,      83556,   85484,   85004, 97677]}

accidents_months_2020 = pd.DataFrame(months_2020)

accidents_months_2020
data = [go.Bar(x=accidents_months_2016['Month of the year - 2016'],

               y=accidents_months_2016['Total Accidents'],

               name='2016'),

        go.Bar(x=accidents_months_2017['Month of the year - 2017'],

               y=accidents_months_2017['Total Accidents'],

               name='2017'),

        go.Bar(x=accidents_months_2018['Month of the year - 2018'],

               y=accidents_months_2018['Total Accidents'],

               name='2018'),

        go.Bar(x=accidents_months_2019['Month of the year - 2019'],

               y=accidents_months_2019['Total Accidents'],

               name='2019'),

        go.Bar(x=accidents_months_2020['Month of the year - 2020'],

               y=accidents_months_2020['Total Accidents'],

               name='2020')

        ]



layout = go.Layout(title='Accidents per month - February 2016 - June 2020',

                   xaxis={'title':'Month of the year'},

                   yaxis={'title':'Number of accidents'},

                   width=1700,

                   height=700)



fig = go.Figure(data=data, layout=layout)



py.iplot(fig)
day = df['Week_day'].value_counts()

day

## Monday is 0, Sunday is 6
d2016 = df['Week_day'].loc[df['Year']==2016].groupby(df['Week_day']).count()

d2016 = pd.DataFrame(d2016)



d2017 = df['Week_day'].loc[df['Year']==2017].groupby(df['Week_day']).count()

d2017 = pd.DataFrame(d2017)



d2018 = df['Week_day'].loc[df['Year']==2018].groupby(df['Week_day']).count()

d2018 = pd.DataFrame(d2018)



d2019 = df['Week_day'].loc[df['Year']==2019].groupby(df['Week_day']).count()

d2019 = pd.DataFrame(d2019)



d2020 = df['Week_day'].loc[df['Year']==2020].groupby(df['Week_day']).count()

d2020 = pd.DataFrame(d2020)
d2016
days_2016 = {'Day of the week - 2016': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],

             'Total Accidents':        [66634,     75322,     74090,        73440,     69640,     20257,     18417]}

days_2016 = pd.DataFrame(days_2016)

days_2016
d2017
days_2017 = {'Day of the week - 2017': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],

             'Total Accidents':        [116952,    125622,    127288,      128576,     130109,   34665,      31217]}

days_2017 = pd.DataFrame(days_2017)

days_2017
d2018
days_2018 = {'Day of the week - 2018': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],

             'Total Accidents':        [153320,   160376,    156840,       150100,     156513,   46868,      40957]}

days_2018 = pd.DataFrame(days_2018)

days_2018
d2019
days_2019 = {'Day of the week - 2019': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],

             'Total Accidents':        [155738,   166964,    164091,       159736,     164243,   62373,      55779]}

days_2019 = pd.DataFrame(days_2019)

days_2019
d2020
days_2020 = {'Day of the week - 2019': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],

             'Total Accidents':        [83294,   85134,      84813,       93049,     99347,   42766,      36458]}

days_2020 = pd.DataFrame(days_2020)

days_2020
data = [go.Bar(x=days_2016['Day of the week - 2016'],

               y=days_2016['Total Accidents'],

               name='2016'),

        go.Bar(x=days_2017['Day of the week - 2017'],

               y=days_2017['Total Accidents'],

               name='2017'),

        go.Bar(x=days_2018['Day of the week - 2018'],

               y=days_2018['Total Accidents'],

               name='2018'),

        go.Bar(x=days_2019['Day of the week - 2019'],

               y=days_2019['Total Accidents'],

               name='2019'),

        go.Bar(x=days_2020['Day of the week - 2019'],

               y=days_2020['Total Accidents'],

               name='2020')]



layout = go.Layout(title='Accidents per day - February 2016 - June 2020',

                   xaxis={'title':'Day of the week'},

                   yaxis={'title':'Number of accidents'},

                   width=1700,

                   height=700)



fig = go.Figure(data=data, layout=layout)



py.iplot(data)
cities = df['City'].value_counts()[df['City'].value_counts()>10000]

cities

## You can change the value on the first line to filter less or more cities with the amount the accidents you want to check
data = [go.Bar(x=cities.index,

               y=cities.values,

               name='Cities')]



layout = go.Layout(title='Accidents per City - 2016 - 2019',

                   yaxis={'title':'Number of accidents'},

                   width=1700,

                   height=700)



fig = go.Figure(data=data, layout=layout)



py.iplot(fig)
county = df['County'].value_counts()[df['County'].value_counts()>10000]

county
data = [go.Bar(x=county.index,

               y=county.values,

               name='County')]



layout = go.Layout(title='Accidents per County - 2016 - 2019',

                   yaxis={'title':'Number of accidents'},

                   width=1700,

                   height=700)



fig = go.Figure(data=data, layout=layout)



py.iplot(fig)
state = df['State'].value_counts()

state
data = [go.Bar(x=state.index,

               y=state.values,

               name='State')]



layout = go.Layout(title='Accidents per State - 2016 - 2019',

                   yaxis={'title':'Number of accidents'},

                   width=1700,

                   height=700)



fig = go.Figure(data=data, layout=layout)



py.iplot(fig)
range_temp = pd.cut(df['Temperature(F)'], 5)

range_temp.unique()
df.loc[ (df['Temperature(F)'] > -30) & (df['Temperature(F)'] <= 10), 'Temperature(F)']=1

df.loc[ (df['Temperature(F)'] > 10) & (df['Temperature(F)'] <= 50), 'Temperature(F)']=2

df.loc[ (df['Temperature(F)'] > 50) & (df['Temperature(F)'] <= 90), 'Temperature(F)']=3

df.loc[ (df['Temperature(F)'] > 90) & (df['Temperature(F)'] <= 130),'Temperature(F)' ]=4

df.loc[  df['Temperature(F)'] > 130, 'Temperature(F)']=5
temp_values = df['Temperature(F)'].value_counts()
data = [go.Bar(x=temp_values.index,

               y=temp_values.values,

               name='Temperature')]



layout = go.Layout(title='Accidents per Temperature - 2016 - 2019',

                   yaxis={'title':'Number of accidents'},

                   width=1200,

                   height=500)



fig = go.Figure(data=data, layout=layout)



fig.add_annotation(x=1,y=0,text="-30-10 F")

fig.add_annotation(x=2,y=0,text="10-50 F")

fig.add_annotation(x=3,y=0,text="50-90 F")

fig.add_annotation(x=4,y=0,text="90-130 F")

fig.add_annotation(x=5,y=0,text=">130 F")



py.iplot(fig)
df.loc[ (df['Humidity(%)'] > 0) & (df['Humidity(%)'] <= 20), 'Humidity(%)']=1

df.loc[ (df['Humidity(%)'] > 20) & (df['Humidity(%)'] <= 40), 'Humidity(%)' ]=2

df.loc[ (df['Humidity(%)'] > 40) & (df['Humidity(%)'] <= 60), 'Humidity(%)']=3

df.loc[ (df['Humidity(%)'] > 60) & (df['Humidity(%)'] <= 80),'Humidity(%)' ]=4

df.loc[  df['Humidity(%)'] > 80, 'Humidity(%)']=5
df['Humidity(%)'].unique()
hum = df['Humidity(%)'].value_counts()

hum
data = [go.Bar(x=hum.index,y=hum.values,name='Humidity')]



layout = go.Layout(title='Accidents according to Humitity',xaxis={'title':'Humidity'}, yaxis={'title':'Accidents'}, width=1200,

                   height=500)



fig = go.Figure(data=data,layout=layout)



fig.add_annotation(x=1,y=0,text="0-20 [%]")

fig.add_annotation(x=2,y=0,text="20-40 [%]")

fig.add_annotation(x=3,y=0,text="40-60 [%]")

fig.add_annotation(x=4,y=0,text="60-80 [%]")

fig.add_annotation(x=5,y=0,text="80-100 [%] F")





py.iplot(fig)
ts = df['Traffic_Signal'].value_counts()
data = [go.Pie(labels=ts.index,

               values=ts.values,

               direction='clockwise')

       ]



layout = go.Layout(title='Near a traffic light?',

                   width=600,

                   height=600)



fig = go.Figure(data=data, layout=layout)



py.iplot(fig)
ss = df['Sunrise_Sunset'].value_counts()
data = [go.Pie(labels=ss.index,

               values=ss.values,

               direction='clockwise')

       ]



layout = go.Layout(title='Day or Night?',

                   width=600,

                   height=600)



fig = go.Figure(data=data, layout=layout)



py.iplot(fig)