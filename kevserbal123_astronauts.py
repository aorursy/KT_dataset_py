# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# word cloud library
from wordcloud import WordCloud

# matplotlib
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
astronauts=pd.read_csv("../input/astronauts.csv")
astronauts.head(5)
astronauts.info()
astronauts['Military Rank'].value_counts()
#Space walks and Space flights according to years

list_of_years=list(astronauts['Year'].unique())
Space_flights_per_year=[]
Space_walks_per_year=[]

#Space flights according to years
for i in list_of_years:
    filter_i=astronauts['Year']==i
    filtered=astronauts[filter_i]
    Space_flights_per_year.append(sum(filtered['Space Flights']))

#Space walks according to years
for i in list_of_years:
    filter_i=astronauts['Year']==i
    filtered=astronauts[filter_i]
    Space_walks_per_year.append(sum(filtered['Space Walks']))

Space_flights_per_year=pd.Series(Space_flights_per_year)
Space_walks_per_year=pd.Series(Space_walks_per_year)   
list_of_years=pd.Series(list_of_years)

data_frame=pd.DataFrame({
    "list_of_years":list_of_years,
    "Space_walks_per_year":Space_walks_per_year,
    "Space_flights_per_year":Space_flights_per_year   })



index=(data_frame.list_of_years.sort_values(ascending=False)).index.values
data_frame=data_frame.reindex(index)

trace1=go.Scatter(
          x= data_frame.list_of_years,
          y= data_frame.Space_walks_per_year,
          mode="lines",
          name="Space Walks",
          marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
          text="")

trace2=go.Scatter(
          x= data_frame.list_of_years,
          y= data_frame.Space_flights_per_year,
          mode="lines",
          name="Space Flights",
          marker = dict(color = 'rgba(10, 26,200, 0.8)'),
          text="")

data = [trace1, trace2]
layout = dict(title = 'Space walks and Space flights according to years',
              xaxis= dict(title= 'Space',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
#Space Flight (hr) according to  Graduate Majors  in 1978,1990 and 1996
filtered_for_year=astronauts[(astronauts.Year==1996.0) | (astronauts.Year==1990.0) | (astronauts.Year==1978.0)]
GM_flight_hour=[]
for i in filtered_for_year['Graduate Major'].unique():
    filtered=filtered_for_year[filtered_for_year['Graduate Major']==i]
    GM_flight_hour.append(sum(filtered['Space Flight (hr)']))


most_flying_GMs=filtered_for_year['Graduate Major'].unique()

number_1978_GM=[]
number_1990_GM=[]
number_1996_GM=[]


for i in most_flying_GMs:
    filtered=filtered_for_year[filtered_for_year['Graduate Major']==i]
    f1978=filtered[filtered.Year==1978.0]
    f1990=filtered[filtered.Year==1990.0]
    f1996=filtered[filtered.Year==1996.0]
    number_1978_GM.append(sum(f1978['Space Flight (hr)'] ))
    number_1990_GM.append(sum(f1990['Space Flight (hr)'] ))
    number_1996_GM.append(sum(f1996['Space Flight (hr)'] ))

dfGM=pd.DataFrame({
    "number_1978_GM":number_1978_GM,
    "number_1990_GM":number_1990_GM,
    "number_1996_GM":number_1996_GM,
    "top_15_Graduate_major":most_flying_GMs})

index=(dfGM['number_1996_GM'].sort_values(ascending=False)).index.values
dfGM=dfGM.reindex(index)
index=pd.Series(range(1,len(most_flying_GMs)))

trace1=go.Scatter(
          x= index,
          y= dfGM.number_1978_GM,
          mode="markers",
          name="1978",
          marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
          text=dfGM.top_15_Graduate_major)

trace2=go.Scatter(
          x= index,
          y= dfGM.number_1990_GM,
          mode="markers",
          name="1990",
          marker = dict(color = 'rgba(16, 12, 200, 0.8)'),
          text=dfGM.top_15_Graduate_major)

trace3=go.Scatter(
          x=index,
          y= dfGM.number_1996_GM,
          mode="markers",
          name="1996",
          marker = dict(color = 'rgba(166, 12, 2, 0.8)'),
          text=dfGM.top_15_Graduate_major)


data = [trace1, trace2, trace3]
layout = dict(title = 'Space Flight (hr) of Graduate Majors in 1978,1990 and 1996',
              xaxis= dict(title= 'Garduate Major',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Space Flight (hr)',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)

status=astronauts['Status'].unique()
female=[]
male=[]
space_flights_female=[]
space_flights_male=[]
for i in status:
    status_i=astronauts[astronauts['Status']==i]
    female.append(len(status_i[status_i.Gender=='Female']))
    male.append(len(status_i[status_i.Gender=='Male']))
    space_flights_female.append(sum(status_i[status_i.Gender=='Female']['Space Flights']))
    space_flights_male.append(sum(status_i[status_i.Gender=='Male']['Space Flights']))

space_flights_male=["flights : " + str(i) for i in space_flights_male]
space_flights_female=["flights : " + str(i) for i in space_flights_female]
  


df_Status=pd.DataFrame({
    "female":female,
    "male":male,
    "space_flights_female":space_flights_female,
    "space_flights_male":space_flights_male,
    "status":status})    


trace1=go.Bar(
          x= df_Status.status,
          y= df_Status.female,
  
          name="female",
          marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
          text=df_Status.space_flights_female)

trace2=go.Bar(
          x= df_Status.status,
          y= df_Status.male,
         
          name="male",
          marker = dict(color = 'rgba(16, 0, 200, 0.8)'),
          text=df_Status.space_flights_male )
data = [trace1, trace2]
layout = go.Layout(barmode = "group")
fig = go.Figure(data = data, layout = layout)
iplot(fig)

#pie chart of military rank in 1978
astronauts['Military Rank'].fillna("No military rank", inplace=True)
df1978=astronauts[astronauts.Year==1978.0]
labels=pd.Series( df1978['Military Rank'].unique())
military_rank_percentage=[(len(df1978[df1978['Military Rank']==i ])/(len(df1978)))*100 for i in labels]

fig = {
  "data": [
    {
      "values": military_rank_percentage,
      "labels": labels,
      "domain": {"x": [0, .5]},
      "name": "Number of Military Rank in 1978",
      "hoverinfo":"label+percent+name",
      "hole": .3,
      "type": "pie"
    },],
  "layout": {
        "title":"Rate of Military Rank in 1978",
        "annotations": [
            { "font": { "size": 20},
              "showarrow": False,
              "text": "Number of Military Rank",
                "x": 0.20,
                "y": 1
            },
        ]
    }
}
iplot(fig)





