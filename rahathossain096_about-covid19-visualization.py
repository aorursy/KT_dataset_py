import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import plotly.express as px

%matplotlib inline

plt.style.use('fivethirtyeight')

pd.set_option('display.max_rows', None)
df1 = pd.read_csv('../input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv')



df2 = pd.read_csv('../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv')



df3 = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')



df4 = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')



df5 = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')



df6 = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')
df1.head(2)
df2.head(2)
df3.head(2)
df4.head(2)
df5.head(2)
df6.head(2)
df3.info()
print('Columns :\n',df3.columns.to_list())
print('Unique Values :\n',df3.nunique())
total = df3.isnull().sum()

#Percent = [(i,df3[i].isna().mean()*100) for i in df3]

percent = (total/df3.isnull().count())*100

#nan = pd.DataFrame(Percent)



missing_data = pd.concat([total,percent], axis = 1, keys=['Total', 'Percent'])

missing_data
df3["Province/State"]= df3["Province/State"].fillna('Unknown')

df3.head()
df3['ActiveCase'] = df3['Confirmed'] - df3['Deaths'] - df3['Recovered']

df3.head(20)
Df3 = df3[df3['ObservationDate'] == max(df3['ObservationDate'])].reset_index()

Df3.head()
labels = Df3['Country/Region']



fig = make_subplots(rows = 1, cols = 2, specs = [[{'type':'domain'}, {'type':'domain'}]])

fig.add_trace(go.Pie(labels = labels, values = Df3['Confirmed'], name = 'Country/Region'),

              1, 1)

fig.add_trace(go.Pie(labels = labels, values = Df3['ActiveCase'], name = 'Country/Region'),

              1, 2)



#fig.update_traces(hole=.2, hoverinfo=Df3['Province/State'])

fig.update_traces(textposition = 'inside', hole = 0.45,textinfo = 'percent+label')



fig.update_layout(

    title_text = "Confirmed case and Active case in countrywise",

    # Add annotations in the center of the donut pies.

    annotations=[dict(text = 'Confirmed', x = 0.13, y = 0, font_size = 20, showarrow = False),

                 dict(text = 'Active', x = 0.87, y = 0, font_size = 20, showarrow = False)]

)

fig.show()
DataWorldwide = Df3.groupby(["ObservationDate"])["Confirmed","ActiveCase","Recovered","Deaths"].sum().reset_index()

DataWorldwide
fig = px.pie(DataWorldwide, values = DataWorldwide.loc[0, ["ActiveCase","Recovered","Deaths"]],

             names = ["Active cases","Recovered","Deaths"],color = ["ActiveCase","Recovered","Deaths"],

             color_discrete_map = {'ActiveCase':'darkblue',

                                 'Recovered':'cyan',

                                 'Deaths':'royalblue'},

             title='Total cases : '+str(DataWorldwide["Confirmed"][0]))



fig.update_traces(textposition = 'inside',textinfo='percent+label')

fig.show()
DataByDate = df3.groupby(["ObservationDate"])["Confirmed","ActiveCase","Recovered","Deaths"].sum().reset_index()

DataByDate.head()
fig = go.Figure()

fig.add_trace(go.Scatter(x = DataByDate['ObservationDate'],

                         y = DataByDate['Confirmed'],

                         mode = 'lines',

                         name = 'Confirmed cases',

                         opacity = 0.8))



fig.add_trace(go.Scatter(x = DataByDate['ObservationDate'],

                         y = DataByDate['ActiveCase'],

                         mode = 'lines',

                         name = 'Active cases',

                         line = dict( dash='dot'),

                         opacity = 0.8))

fig.add_trace(go.Scatter(x = DataByDate['ObservationDate'],

                         y = DataByDate['Deaths'],

                         name = 'Deaths',

                         marker_color = 'black'

                         ,mode = 'lines',

                         line = dict( dash='dot'),

                         opacity = 0.8))

fig.add_trace(go.Scatter(x = DataByDate['ObservationDate'],

                         y = DataByDate['Recovered'],

                         mode = 'lines',

                         name = 'Recovered cases',

                         marker_color = 'green',

                         opacity = 0.8))

fig.update_layout(

    title = 'Evolution of cases over time in Worldwide',

)



fig.show()
country = df3[df3.ObservationDate == df3.ObservationDate.max()].groupby('Country/Region').sum().reset_index()

country.head(2)
country['DeathRate'] = (country['Deaths']/country['Confirmed'])*100

country['RecoveryRate'] = (country['Recovered']/country['Confirmed'])*100

country.head(3)
#DeathRate

SortedDR = country.sort_values('DeathRate',ascending = False)



fig = go.Figure(data = [go.Bar(

            x = SortedDR['Country/Region'][0:15], y = SortedDR['DeathRate'][0:15],

            textposition = 'auto',

            marker_color = 'rgb(199, 71, 60)',

        )])

fig.update_layout(

    title = 'Death Rate top 15 Countries',

    xaxis_title = "Countries",

    yaxis_title = "Death Rate",

)



fig.show()
#RecoveryRate

SortedRR = country.sort_values('RecoveryRate',ascending = False)

fig = go.Figure(data = [go.Bar(

            x = SortedRR['Country/Region'][0:15], y = SortedRR['RecoveryRate'][0:15],

            textposition = 'auto',

            marker_color = 'rgb(47, 150, 39)',

        )])

fig.update_layout(

    title = 'Recovery Rate of Top 15 Countries',

    xaxis_title = "Countries",

    yaxis_title = "Recovery Rate",

)





fig.show()
fig = px.pie(country, values = 'RecoveryRate', names = 'Country/Region',

             title = 'Countrywise Recovery Rate Cases', labels = {'Province/State':'Province/State'})

fig.update_traces(textposition = 'inside')

fig.show()
Df3["world"] = "world" # in order to have a single root node

fig = px.treemap(Df3, path = ['world' , 'Country/Region', 'Province/State'], color = 'Recovered' ,color_continuous_scale = px.colors.sequential.Viridis,title = 'Recovery rates', values = 'Recovered')

fig.show()
Data_BD = df3[(df3['Country/Region'] == 'Bangladesh') ].reset_index(drop=True)

Data_BD
fig = go.Figure()

fig.add_trace(go.Scatter(x = Data_BD['ObservationDate'],

                         y = Data_BD['Confirmed'],

                         mode = 'lines',

                         name = 'Confirmed cases',

                         opacity = 0.8))



fig.add_trace(go.Scatter(x = Data_BD['ObservationDate'],

                         y = Data_BD['ActiveCase'],

                         mode = 'lines',

                         name = 'Active cases',

                         line = dict( dash = 'dot'),

                         opacity = 0.8))

fig.add_trace(go.Scatter(x = Data_BD['ObservationDate'],

                         y = Data_BD['Deaths'],

                         name = 'Deaths',

                         marker_color = 'black'

                         ,mode = 'lines',

                         line = dict( dash = 'dot'),

                         opacity = 0.8))

fig.add_trace(go.Scatter(x = Data_BD['ObservationDate'],

                         y = Data_BD['Recovered'],

                         mode = 'lines',

                         name = 'Recovered cases',

                         marker_color = 'green',

                         opacity = 0.8))

fig.update_layout(

    title='Coronavirus Spread Over Time in Bangladesh',

)



fig.show()
plt.figure(figsize=(12,6))

plt.plot(Data_BD["Confirmed"],marker = "o",label = "Confirmed Cases")

plt.plot(Data_BD["Recovered"],marker = "*",label = "Recovered Cases")

plt.plot(Data_BD["Deaths"],marker = "^",label = "Death Cases")

plt.ylabel("Number of Patients")

plt.xlabel("Date")

plt.xticks(rotation = 90)

plt.title("Coronavirus Spread Over Time in Bangladesh")

plt.legend();
BD_last = Data_BD[Data_BD['ObservationDate'] == max(Data_BD['ObservationDate'])].reset_index()

BD_last
fig = px.pie(BD_last, values = BD_last.loc[0, ["ActiveCase","Recovered","Deaths"]],

             names = ["Active cases","Recovered","Deaths"],color = ["ActiveCase","Recovered","Deaths"],

             color_discrete_map = {'ActiveCase':'darkblue',

                                 'Recovered':'cyan',

                                 'Deaths':'royalblue'},

             title = 'Total cases : '+str(BD_last["Confirmed"][0]))



fig.update_traces(textposition = 'inside',textinfo = 'percent+label')

fig.show()