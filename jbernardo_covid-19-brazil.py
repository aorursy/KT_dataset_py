import numpy as np

import pandas as pd

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot, plot

import plotly.express as px

from plotly.subplots import make_subplots



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.



#Loading database

COVID19_BR = pd.read_csv("../input/corona-virus-brazil/brazil_covid19.csv")

#Transforming data

COVID19_BR.date = pd.to_datetime(COVID19_BR.date)

COVID19_BR.hour = pd.to_datetime(COVID19_BR.hour, format='%H:%M').dt.time

COVID19_BR["Week_Number"] = COVID19_BR.date.dt.week



#Removing duplicated lines (when updated more than once a day)

COVID19_BR = COVID19_BR.drop_duplicates(subset=["date","state"], keep = "last")



#Rename columns

COVID19_BR.columns = ['Date', 'Hour', 'State', 'Suspects', 'Negative', 'Confirmed', "Deaths","Week_Number"]



#reading data infos

COVID19_BR.info()
temp = COVID19_BR.groupby('State', as_index=False)["Suspects","Negative","Confirmed","Deaths"].max()

msg = """

This database contains the cases published by Ministry of Health over the time since the first case of COVID-19 in Brazil.

The first record in database was on """ + str(COVID19_BR.Date.dt.date.min()) + """ 

and the last record updated was on """ + str(COVID19_BR.Date.dt.date.max()) + """

Until now, Ministry of Healthy has recorded """ + str(temp.Suspects.sum()) + """ suspects, the number of negative cases is """ + str(temp.Negative.sum()) + """, """ + str(temp.Confirmed.sum()) + """ were confirmed and """ + str(temp.Deaths.sum()) + """ deaths have been reported.

"""

print(msg)
temp = COVID19_BR.groupby('Date', as_index=False)['Confirmed'].sum()

temp = temp[(temp.Confirmed>0)]

x1 = temp.Date

y1 = temp.Confirmed

temp = COVID19_BR.groupby('Date', as_index=False)['Deaths'].sum()

temp = temp[(temp.Deaths>0)]

x2 = temp.Date

y2 = temp.Deaths



fig = make_subplots(rows=1, cols=2)



fig.add_trace(    

    go.Scatter(x = x1, y = y1, name = y1.name,

                         line=dict(color='green', width=2)),

    row=1, col=1

)



fig.add_trace(

    go.Scatter(x = x2, y = y2, name = y2.name,

                         line=dict(color='darkred', width=2)),

    row=1, col=2

)



fig.update_layout({

"plot_bgcolor": "rgba(0, 0, 0, 0)",

"paper_bgcolor": "rgba(0, 0, 0, 0)",

})



fig.update_layout(title_text="Number of cases - COVID-19 in Brazil")



fig.show()

print("Data validation: https://veja.abril.com.br/saude/a-epidemia-de-coronavirus-no-brasil-em-tempo-real/")
print("Cases per Brazilian State - sorted by Deaths")

temp = COVID19_BR.groupby('State', as_index=False)["Suspects","Negative","Confirmed","Deaths"].max()

temp.sort_values("Deaths",ascending = False).style.background_gradient(cmap='Reds')
col = "Suspects"

temp = COVID19_BR.groupby('State', as_index=False)[col].max()

temp = temp[temp[col]>0]



print("Cases per Brazilian State -  "+temp[col].name+" Cases")



fig = px.bar(temp, x="State", y=col, text=col)

fig.update_layout(xaxis={'categoryorder':'total descending'})

fig.update_layout(

    title= "COVID-19 - Number of " + temp[col].name + " cases in Brazilian States")

fig.update_layout({

"plot_bgcolor": "rgba(0, 0, 0, 0)",

"paper_bgcolor": "rgba(0, 0, 0, 0)",

})



fig.show()
col = "Negative"

temp = COVID19_BR.groupby('State', as_index=False)[col].max()

temp = temp[temp[col]>0]



print("Cases per Brazilian State -  "+temp[col].name+" Cases")



fig = px.bar(temp, x="State", y=col, text=col)

fig.update_layout(xaxis={'categoryorder':'total descending'})

fig.update_layout(

    title= "COVID-19 - Number of " + temp[col].name + " cases in Brazilian States")

fig.update_layout({

"plot_bgcolor": "rgba(0, 0, 0, 0)",

"paper_bgcolor": "rgba(0, 0, 0, 0)",

})



fig.show()

col = "Confirmed"

temp = COVID19_BR.groupby('State', as_index=False)[col].max()

temp = temp[temp[col]>0]



print("Cases per Brazilian State -  "+temp[col].name+" Cases")



fig = px.bar(temp, x="State", y=col, text=col)

fig.update_layout(xaxis={'categoryorder':'total descending'})

fig.update_layout(

    title= "COVID-19 - Number of " + temp[col].name + " cases in Brazilian States")

fig.update_layout({

"plot_bgcolor": "rgba(0, 0, 0, 0)",

"paper_bgcolor": "rgba(0, 0, 0, 0)",

})



fig.show()

col = "Deaths"

temp = COVID19_BR.groupby('State', as_index=False)[col].max()

temp = temp[temp[col]>0]



print("Cases per Brazilian State -  "+temp[col].name+" Cases")



fig = px.bar(temp, x="State", y=col, text=col)

fig.update_layout(xaxis={'categoryorder':'total descending'})

fig.update_layout(

    title= "COVID-19 - Number of " + temp[col].name + " cases in Brazilian States")

fig.update_layout({

"plot_bgcolor": "rgba(0, 0, 0, 0)",

"paper_bgcolor": "rgba(0, 0, 0, 0)",

})



fig.show()

temp = COVID19_BR.groupby('Date', as_index=False)['Suspects', 'Negative', 'Confirmed', 'Deaths'].sum()

temp.head()

x1 = temp.Date

y1 = temp.Suspects

y2 = temp.Negative

y3 = temp.Confirmed

y4 = temp.Deaths



trace1 = go.Scatter(x = x1, y = y1, name = y1.name,

                         line=dict(color='orange', width=2))

trace2 = go.Scatter(x = x1, y = y2, name = y2.name,

                         line=dict(color='blue', width=2, dash='dot'))

trace3 = go.Scatter(x = x1, y = y3, name = y3.name,

                         line=dict(color='green', width=2))

trace4 = go.Scatter(x = x1, y = y4, name = y4.name,

                         line=dict(color='darkred', width=2))

data = [trace1, trace2, trace3, trace4]

layout = dict(title = 'Cases over time - COVID-19 in Brazil',

              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Number of cases',ticklen= 5,zeroline= False),

              plot_bgcolor='white'

             )

fig = dict(data = data, layout = layout)



iplot(fig)
col = "Suspects"



fig = go.Figure()

for name, group in COVID19_BR.groupby('State'):

    #trace = go.Histogram()

    trace = go.Scatter(x = group.Date, y = group[col])

    trace.name = name

#    trace.x = group.Deaths

    fig.add_trace(trace)

#fig.update_xaxes(title_text = "Date")

#fig.update_yaxes(title_text = "Number of cases")



fig.update_layout(

    title="Cases over time - COVID-19 - " + str(col) + " cases in Brazilian States",

    xaxis_title="Date",

    yaxis_title="Number of cases - " + temp[col].name

)



fig.update_layout({

"plot_bgcolor": "rgba(0, 0, 0, 0)",

"paper_bgcolor": "rgba(0, 0, 0, 0)",

})

iplot(fig)

print("You are able to hide some lines here")
col = "Negative"



fig = go.Figure()

for name, group in COVID19_BR.groupby('State'):

    #trace = go.Histogram()

    trace = go.Scatter(x = group.Date, y = group[col])

    trace.name = name

#    trace.x = group.Deaths

    fig.add_trace(trace)

#fig.update_xaxes(title_text = "Date")

#fig.update_yaxes(title_text = "Number of cases")



fig.update_layout(

    title="Cases over time - COVID-19 - " + str(col) + " cases in Brazilian States",

    xaxis_title="Date",

    yaxis_title="Number of cases - " + temp[col].name

)



fig.update_layout({

"plot_bgcolor": "rgba(0, 0, 0, 0)",

"paper_bgcolor": "rgba(0, 0, 0, 0)",

})

iplot(fig)

print("You are able to hide some lines here")
col = "Confirmed"



fig = go.Figure()

for name, group in COVID19_BR.groupby('State'):

    #trace = go.Histogram()

    trace = go.Scatter(x = group.Date, y = group[col])

    trace.name = name

#    trace.x = group.Deaths

    fig.add_trace(trace)

#fig.update_xaxes(title_text = "Date")

#fig.update_yaxes(title_text = "Number of cases")



fig.update_layout(

    title="Cases over time - COVID-19 - " + str(col) + " cases in Brazilian States",

    xaxis_title="Date",

    yaxis_title="Number of cases - " + temp[col].name

)



fig.update_layout({

"plot_bgcolor": "rgba(0, 0, 0, 0)",

"paper_bgcolor": "rgba(0, 0, 0, 0)",

})

iplot(fig)

print("You are able to hide some lines here")
col = "Deaths"



fig = go.Figure()

for name, group in COVID19_BR.groupby('State'):

    #trace = go.Histogram()

    trace = go.Scatter(x = group.Date, y = group[col])

    trace.name = name

#    trace.x = group.Deaths

    fig.add_trace(trace)

#fig.update_xaxes(title_text = "Date")

#fig.update_yaxes(title_text = "Number of cases")



fig.update_layout(

    title="Cases over time - COVID-19 - " + str(col) + " cases in Brazilian States",

    xaxis_title="Date",

    yaxis_title="Number of cases - " + temp[col].name

)



fig.update_layout({

"plot_bgcolor": "rgba(0, 0, 0, 0)",

"paper_bgcolor": "rgba(0, 0, 0, 0)",

})

iplot(fig)

print("You are able to hide some lines here")
print("Number States with confirmed cases over time")

temp = COVID19_BR[["Date","State","Confirmed"]].groupby("Date", as_index=False)["State"].count()

x1 = temp.Date

y1 = temp.State



trace1 = go.Scatter(x = x1, y = y1, name = y1.name)

data = [trace1]

layout = dict(title = 'Number States with confirmed cases over time - COVID-19 in Brazil',

              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Number of states',ticklen= 5,zeroline= False),

              plot_bgcolor='white'

             )

fig = dict(data = data, layout = layout)



iplot(fig)