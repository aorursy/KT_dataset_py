# 786

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import plotly.graph_objs as go



import plotly as py

from plotly import tools

from plotly.offline import iplot



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dt = pd.read_csv("../input/pakistan-corona-virus-citywise-data/PK COVID-19-30apr.csv", encoding = "ISO-8859-1", parse_dates=["Date"])

print("Data Dimensions are: ", dt.shape)

print(dt.head)
dt.info()
dt['Travel_history'].unique

dt['Travel_history'].fillna('Unknown',  inplace=True)
dt = dt.sort_values('Date')

dt['Deaths']=dt['Deaths'].astype(int)

dt['Cases']=dt['Cases'].astype(int)

dt['Recovered']=dt['Recovered'].astype(int)



dt.loc[dt.Province == "khyber Pakhtunkhwa", "Province"] = "Khyber Pakhtunkhwa"

dt.loc[dt.Travel_history == "Tableegi Jamaat", "Travel_history"] = "Tableeghi Jamaat"
pdc = dt.groupby('Date')['Cases'].sum().reset_index()

pdd = dt.groupby('Date')['Deaths'].sum().reset_index()#.drop('Date', axis=1)

pdr = dt.groupby('Date')['Recovered'].sum().reset_index()#.reset_index()#.drop('Date', axis=1)



p = pd.DataFrame(pdc) 

p['Deaths'] = pdd['Deaths']

p['Recovered'] = pdr['Recovered']



#Cumulative Sum

p['Cum_Cases'] = p['Cases'].cumsum() 

p['Cum_Deaths'] = p['Deaths'].cumsum()

p['Cum_Recovered'] = p['Recovered'].cumsum()



prbind = pd.DataFrame(pdc) 

prbind['Type'] = 'Cases'

prbind.rename(columns={'Cases': 'Count'}, inplace=True)



pdd['Type'] = 'Deaths'

pdd.rename(columns={'Deaths': 'Count'}, inplace=True)

prbind =prbind.append(pdd)



pdr['Type'] = 'Recovered'

pdr.rename(columns={'Recovered': 'Count'}, inplace=True)

prbind =prbind.append(pdr)



del pdc, pdd, pdr

p['Dateofmonth'] = p['Date'].dt.day

p['Month'] = p['Date'].dt.month

p['Week'] = p['Date'].dt.week

p['Dayofweek'] = p['Date'].dt.dayofweek # 0 = monday.

p['Weekdayflg'] = (p['Dayofweek'] // 5 != 1).astype(float)

p['Month'] = p['Date'].dt.month

p['Quarter'] = p['Date'].dt.quarter

p['Dayofyear'] = p['Date'].dt.dayofyear

p.head(10)
fig = go.Figure()



# Add traces

fig.add_trace(go.Scatter(x=p['Date'], y=p['Cases'],

                    mode='lines+markers',

                    name='Cases'))

fig.add_trace(go.Scatter(x=p['Date'], y=p['Deaths'],

                    mode='lines+markers',

                    name='Deaths'))

fig.add_trace(go.Scatter(x=p['Date'], y=p['Recovered'],

                    mode='lines+markers',

                    name='Recoveries'))



fig.show()
fig = go.Figure()



# Add traces

fig.add_trace(go.Scatter(x=p['Date'], y=p['Cum_Cases'],

                    mode='lines+markers',

                    name='Cases'))

fig.add_trace(go.Scatter(x=p['Date'], y=p['Cum_Deaths'],

                    mode='lines+markers',

                    name='Deaths'))

fig.add_trace(go.Scatter(x=p['Date'], y=p['Cum_Recovered'],

                    mode='lines+markers',

                    name='Recoveries'))



fig.show()
px.box(prbind, x='Type', y='Count')
n = dt.groupby(['Date' ,'Province'])['Cases'].sum().reset_index()

px.box(n, y="Cases", color = "Province")
n = dt.groupby(['Date' ,'Travel_history'])['Cases'].sum().reset_index()

px.box(n, y="Cases", color = "Travel_history")
px.scatter(p, x= 'Date', y = 'Cases', trendline = "ols")
n = dt.groupby(['Date' ,'Province'])['Cases'].sum().reset_index()

fig = px.scatter(n, x="Date", y="Cases", color="Province", trendline="lowess")

fig.show()
n = dt.query("Province == 'Sindh' or Province == 'Punjab'")

n = n.groupby(['Date', 'Travel_history' ,'Province'])['Cases'].sum().reset_index()

fig = px.scatter(n, x="Date", y="Cases", facet_col="Province", color="Travel_history", trendline="ols")

fig.show()



results = px.get_trendline_results(fig)

#print(results)



results.query("Province == 'Sindh' or Province == 'Punjab'").px_fit_results.iloc[0].summary()
n = dt.groupby('Date')['Cases'].sum().reset_index()

px.density_contour(dt,x="Date",y="Cases",marginal_x="histogram",marginal_y="histogram")
fig = go.Figure()

x = dt["Date"]

y = dt["Cases"]

fig.add_trace(go.Histogram2dContour(

        x = x,

        y = y,

        colorscale = 'Reds',

        xaxis = 'x',

        yaxis = 'y'

    ))

fig.add_trace(go.Scatter(

        x = x,

        y = y,

        xaxis = 'x',

        yaxis = 'y',

        mode = 'markers',

        marker = dict(

            color = 'black',

            size = 2

        )

    ))
n = dt.groupby(['Date' ,'Province','City', 'Travel_history'])['Cases'].sum().reset_index()

px.scatter(n, x= 'Date', y = 'Cases', color="Travel_history", size ="Cases",  hover_data=['Province','City'])
n = dt.groupby(['Date' ,'Province'])['Cases'].sum().reset_index()

px.line(n, x='Date', y='Cases', color='Province')
# cases by date

cases_perdate = np.asarray(dt.groupby('Date')['Cases'].sum())



# 7 day moving average of cases by date

ms_average = pd.Series(cases_perdate).rolling(window=7).mean()

#ms_average = np.asarray(ms_average.drop(ms_average.index[0:6]))

ms_average = np.round(ms_average, 0)



dt_dates = np.arange('2019-02', '2019-05', dtype='datetime64[D]')

dt_range = dt_dates[25:81]



trace_date = go.Scatter(

             x = dt_dates,

             y = cases_perdate,

             mode = 'lines',

             name = 'Cases',

             line = dict(

                 color = 'rgb(215, 0, 0)',

                 width = 3)

             )



trace_mean = go.Scatter(

             x = dt_range,

             y = ms_average,

             mode = 'lines',

             name = 'Average',

             line = dict(

                 color = 'rgb(215, 0, 0)',

                 width = 5),

             opacity = 0.33

             )



layout = go.Layout(

         title = 'Covid Cases by Date in Pakistan <br>'

                 '<sub>Hover & Rescale Plot to Desired Dates</sub>',

         showlegend = False,

         xaxis = dict(

             rangeslider = dict(thickness = 0.05),

             type = 'date',

             showline = True,

             showgrid = False

         ),

         yaxis = dict(

             range = [0, 56],

             autotick = False,

             tick0 = 10,

             dtick = 10,

             showline = True,

             showgrid = False)

         )



data = [trace_date, trace_mean]

figure = dict(data = data, layout = layout)

iplot(figure)