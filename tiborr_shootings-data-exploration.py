# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd



import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
# Read input data

df16=pd.read_csv('../input/2016.csv',encoding = "ISO-8859-1")

df16.dropna(inplace=True)



# Removing irrelevant information

df16 = df16.drop(["lawenforcementagency", "name", "streetaddress", "year"], 1)



# Arab-American removed because of its lack of 

# occurence and less outside data for that ethnicity

df16 = df16[df16['raceethnicity'] != "Arab-American"]

df16 = df16[df16['raceethnicity'] != "Unknown"]



# Get counts of killings by race and sort it

raceCounts16 = df16["raceethnicity"].value_counts().sort_index()



df16.head()
races = sorted(df16["raceethnicity"].unique())



racePopPercents = dict(zip(races, [4.9, 12.2, 16.3, 0.7, 63.7]))

for race in races:

    raceCounts16.ix[[race]] = raceCounts16.ix[[race]]/racePopPercents[race]
gbArmedAndRace = df16.groupby(["raceethnicity", "armed"])["uid"].count()

armedByRaceData16 = []

for race in races:

    armedByRaceData16.append(gbArmedAndRace[race]["No"]/gbArmedAndRace[race].sum()*3)
sns.set_style("ticks")

plt.figure(figsize=(12, 6))

ax = sns.barplot(x=races, y=raceCounts16)

sns.set(font_scale=1.5)

ax.set(xlabel='Race (width of bar represents percentage of non-armed killings)', ylabel='Police shootings proportional to population')

sns.plotting_context("notebook", font_scale=10)

for bar,newwidth in zip(ax.patches,armedByRaceData16):

    x = bar.get_x()

    width = bar.get_width()

    centre = x+width/2.



    bar.set_x(centre-newwidth/2.)

    bar.set_width(newwidth)
causeByRace = df16.groupby(["raceethnicity", "classification"])["uid"].count()

numByRace = list(df16["raceethnicity"].value_counts().sort_index())



gunshot = []

vehicle = []

custody = []

taser = []

other = []

for race in races:

    gunshot.append(causeByRace[race]["Gunshot"])

    if "Struck by vehicle" in causeByRace[race]:

        vehicle.append(causeByRace[race]["Struck by vehicle"])

    else:

        vehicle.append(0)

    if "Death in custody" in causeByRace[race]:

        custody.append(causeByRace[race]["Death in custody"])

    else:

        custody.append(0)

    if "Taser" in causeByRace[race]:

        taser.append(causeByRace[race]["Taser"])

    else:

        taser.append(0)

    if "Other" in causeByRace[race]:

        other.append(causeByRace[race]["Other"])

    else:

        other.append(0)



getPercent = lambda x,y: round(x/y*100, 2)

gunshot = list(map(getPercent, gunshot, numByRace))

vehicle = list(map(getPercent, vehicle, numByRace))

custody = list(map(getPercent, custody, numByRace))

taser = list(map(getPercent, taser, numByRace))

other = list(map(getPercent, other, numByRace))



trace0 = go.Bar(

    x=races,

    y=gunshot,

    name='Gunshot',

    marker=dict(

        color='rgb(49,130,189)'

    )

)

trace1 = go.Bar(

    x=races,

    y=vehicle,

    name='Struck by vehicle',

    marker=dict(

        color='rgb(204,204,204)',

    )

)

trace2 = go.Bar(

    x=races,

    y=custody,

    name='Death in custody',

    marker=dict(

        color='rgb(180,0,0)',

    )

)

trace3 = go.Bar(

    x=races,

    y=taser,

    name='Taser',

    marker=dict(

        color='rgb(34,139,34)',

    )

)

trace4 = go.Bar(

    x=races,

    y=taser,

    name='Other',

    marker=dict(

        color='rgb(0,255,255)',

    )

)



data = [trace1, trace2, trace3, trace4]

layout = go.Layout(

    title="Percentage of police killings by cause of death",

    xaxis=dict(tickangle=-30, title="Race"),

    yaxis=dict(title="Percentage of killings"),

    barmode='group',

    width=600,

    height=500

)



fig = go.Figure(data=data, layout=layout)

iplot(fig)