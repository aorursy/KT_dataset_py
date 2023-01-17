# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly import tools
import random

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/athlete_events.csv")
df.head()
df.info()
df.columns
df.Medal.unique()
dataMedal = df[(df.Medal == 'Gold') | (df.Medal == 'Silver') | (df.Medal == 'Bronze')]
dataMedal["CountMedal"] = 1
dataMedal.head()
world = dataMedal.groupby("NOC").sum()['CountMedal'].sort_values(ascending=False)
plt.figure(figsize=(20,15))
data = [ dict(
        type = 'choropleth',
        locations = world.index,
        z = world.values,
        text = world.index,
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            title = 'Medals'),
      ) ]

layout = dict(
    autosize=True,
    width=1000,
    height=600,
    title = 'How Many Olympic Medals Each Country Has Won',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        showland = True,
        landcolor = "#DFDFD0",
        projection = dict(
            type = 'miller'
        )
    )
)

fig = dict( data=data, layout=layout )
iplot(fig)
most30MedalCountries = dataMedal.Team.value_counts()[:31]
data = [go.Bar(
        x = most30MedalCountries.index,
        y = most30MedalCountries.values,
       
)]

layout = go.Layout(
    title = '30 Countries Have Most Medals' )

fig = go.Figure(data=data, layout = layout)
iplot(fig)
dataOfGoldMedals = dataMedal[dataMedal.Medal == 'Gold']
dataOfSilverMedals = dataMedal[dataMedal.Medal == 'Silver']
dataOfBronzeMedals = dataMedal[dataMedal.Medal == 'Bronze']

trace1 = go.Scatter(x=dataOfGoldMedals.Team.value_counts().index[:21],
                  y=dataOfGoldMedals.Team.value_counts().values[:21],
                  line = dict(color="yellow", width=4),
                  name = "Number of Gold Medals",
                   xaxis = 'x3',
                   yaxis = 'y3')
trace2 = go.Scatter(x=dataOfSilverMedals.Team.value_counts().index[:21],
                  y=dataOfSilverMedals.Team.value_counts().values[:21],
                  line = dict(color="gray", width=4),
                  name = "Number of Silver Medals",
                   xaxis = 'x1',
                   yaxis = 'y1')
trace3 = go.Scatter(x=dataOfBronzeMedals.Team.value_counts().index[:21],
                  y=dataOfBronzeMedals.Team.value_counts().values[:21],
                  line = dict(color="red", width=4),
                  name = "Number of Bronze Medals",
                   xaxis = 'x2',
                   yaxis = 'y2')
data = [trace1, trace2, trace3]
layout = go.Layout(
    xaxis=dict(
        domain=[0, 0.45],
        anchor='y1'
    ),
    yaxis=dict(
        domain=[0, 0.45],
        anchor='x1'
    ),
    xaxis2=dict(
        domain=[0.55, 1],
        anchor='y2'
    ),
    xaxis3=dict(
        domain=[0, 0.45],
        anchor='y3'
    ),
    yaxis2=dict(
        domain=[0, 0.45],
        anchor='x2'
    ),
    yaxis3=dict(
        domain=[0.55, 1]
    )
    )
fig = go.Figure(data=data, layout=layout)
iplot(fig)
print(df.Sport.unique())
print(len(df.Sport.unique()))
top10Athletics = dataMedal[dataMedal.Sport == "Athletics"].Team.value_counts()[:11]
plt.figure(figsize=(20,6))
sns.barplot(x=top10Athletics.values, y=top10Athletics.index)
plt.xlabel("Total Medals")
plt.title("Top 10 Countries Have Most Medals in Athletics")
plt.xticks(np.arange(0,1150,50))
plt.show()
top10AthleticsGold = dataMedal[(dataMedal.Sport == "Athletics") & (dataMedal.Medal == "Gold")].Team.value_counts()[:11]
sns.set(style="dark", context="talk")
plt.figure(figsize=(20,6))
sns.barplot(x=top10AthleticsGold.values, y=top10AthleticsGold.index, palette = "YlOrBr_r")
plt.xlabel("Total Gold Medals")
plt.title("Top 10 Countries Have Most Gold Medals in Athletics")
plt.xticks(np.arange(0,650,50))
plt.show()
top10Gymnastics = dataMedal[dataMedal.Sport == "Gymnastics"].Team.value_counts()[:11]
plt.figure(figsize=(20,6))
sns.barplot(x=top10Gymnastics.index,y=top10Gymnastics.values,palette = "rocket")
plt.title("Top 10 Countries Have Most Medals in Gymnastics")
plt.ylabel("Total Medals")
plt.xticks(rotation = 45, ha="right")
plt.show()

top10GymnasticsGold = dataMedal[(dataMedal.Sport == "Gymnastics") & (dataMedal.Medal == "Gold")].Team.value_counts()[:11]
plt.figure(figsize=(20,6))
sns.barplot(x=top10GymnasticsGold.index, y=top10GymnasticsGold.values, palette = "BrBG")
plt.ylabel("Total Gold Medals")
plt.xticks(rotation=40, ha="right")
plt.title("Top 10 Countries Have Most Gold Medals in Gymnastics")
plt.show()
top10Swimming = dataMedal[dataMedal.Sport == "Swimming"].Team.value_counts()[:11]
plt.figure(figsize=(20,6))
sns.barplot(x=top10Swimming.index,y=top10Swimming.values,palette = "PuBu_r")
plt.title("Top 10 Countries Have Most Medals in Swimming")
plt.ylabel("Total Medals")
plt.xticks(rotation = 45, ha="right")
plt.show()
top10SwimmingGold = dataMedal[(dataMedal.Sport == "Swimming") & (dataMedal.Medal == "Gold")].Team.value_counts()[:11]
plt.figure(figsize=(20,6))
sns.barplot(x=top10SwimmingGold.index,y=top10SwimmingGold.values,palette = "YlOrRd_r")
plt.title("Top 10 Countries Have Most Gold Medals in Swimming")
plt.ylabel("Total Gold Medals")
plt.xticks(rotation = 45, ha="right")
plt.show()
top10Shooting = dataMedal[dataMedal.Sport == "Shooting"].Team.value_counts()[:11]
plt.figure(figsize=(20,6))
sns.barplot(x=top10Shooting.index,y=top10Shooting.values,palette = "Reds_r")
plt.title("Top 10 Countries Have Most Medals in Shooting")
plt.ylabel("Total Medals")
plt.xticks(rotation = 45, ha="right")
plt.show()
top10ShootingGold = dataMedal[(dataMedal.Sport == "Shooting") & (dataMedal.Medal == "Gold")].Team.value_counts()[:11]
plt.figure(figsize=(20,6))
sns.barplot(x=top10ShootingGold.index,y=top10ShootingGold.values,palette = "BuPu_r")
plt.title("Top 10 Countries Have Most Gold Medals in Shooting")
plt.ylabel("Total Gold Medals")
plt.xticks(rotation = 45, ha="right")
plt.show()
top10Cycling = dataMedal[dataMedal.Sport == "Cycling"].Team.value_counts()[:11]
plt.figure(figsize=(20,6))
sns.barplot(x=top10Cycling.index,y=top10Cycling.values,palette = "Reds_r")
plt.title("Top 10 Countries Have Most Medals in Shooting")
plt.ylabel("Total Medals")
plt.xticks(rotation = 45, ha="right")
plt.show()
top10CyclingGold = dataMedal[(dataMedal.Sport == "Cycling") & (dataMedal.Medal == "Gold")].Team.value_counts()[:11]
plt.figure(figsize=(20,6))
sns.barplot(x=top10CyclingGold.index,y=top10CyclingGold.values,palette = "Oranges_r")
plt.title("Top 10 Countries Have Most Gold Medals in Cycling")
plt.ylabel("Total Gold Medals")
plt.xticks(rotation = 45, ha="right")
plt.show()
top10Wrestling = dataMedal[dataMedal.Sport == "Wrestling"].Team.value_counts()[:11]
plt.figure(figsize=(20,6))
sns.barplot(x=top10Wrestling.index,y=top10Wrestling.values,palette = "Spectral")
plt.title("Top 10 Countries Have Most Medals in Wrestling")
plt.ylabel("Total Medals")
plt.xticks(rotation = 45, ha="right")
plt.show()
top10WrestlingGold = dataMedal[(dataMedal.Sport == "Wrestling") & (dataMedal.Medal == "Gold")].Team.value_counts()[:11]
plt.figure(figsize=(20,6))
sns.barplot(x=top10WrestlingGold.index,y=top10WrestlingGold.values,palette = "Greens_r")
plt.title("Top 10 Countries Have Most Gold Medals in Wrestling")
plt.ylabel("Total Gold Medals")
plt.xticks(rotation = 45, ha="right")
plt.show()
top10Football = dataMedal[dataMedal.Sport == "Football"].Team.value_counts()[:11]
plt.figure(figsize=(20,6))
sns.barplot(x=top10Football.index,y=top10Football.values,palette = "PRGn_r")
plt.title("Top 10 Countries Have Most Medals in Football")
plt.ylabel("Total Medals")
plt.xticks(rotation = 45, ha="right")
plt.show()
top10FootballGold = dataMedal[(dataMedal.Sport == "Football") & (dataMedal.Medal == "Gold")].Team.value_counts()[:11]
plt.figure(figsize=(20,6))
sns.barplot(x=top10FootballGold.index,y=top10FootballGold.values,palette = "RdBu")
plt.title("Top 10 Countries Have Most Gold Medals in Football")
plt.ylabel("Total Gold Medals")
plt.xticks(rotation = 45, ha="right")
plt.show()
top10Basketball = dataMedal[dataMedal.Sport == "Basketball"].NOC.value_counts()[:11]
plt.figure(figsize=(20,6))
sns.barplot(x=top10Basketball.index,y=top10Basketball.values,palette = "RdBu_r")
plt.title("Top 10 Countries Have Most Medals in Basketball")
plt.ylabel("Total Medals")
plt.xticks(rotation = 45, ha="right")
plt.show()
top5BasketballGold = dataMedal[(dataMedal.Sport == "Basketball") & (dataMedal.Medal == "Gold")].NOC.value_counts()[:6]
plt.figure(figsize=(20,6))
sns.barplot(x=top5BasketballGold.index,y=top5BasketballGold.values,palette = "Reds_r")
plt.title("Top 5 Countries Have Most Gold Medals in Basketball")
plt.ylabel("Total Gold Medals")
plt.xticks(rotation = 45, ha="right")
plt.show()
top10Sailing = dataMedal[dataMedal.Sport == "Sailing"].NOC.value_counts()[:11]
plt.figure(figsize=(20,6))
sns.barplot(x=top10Sailing.index,y=top10Sailing.values,palette = "BuPu_r")
plt.title("Top 10 Countries Have Most Medals in Sailing")
plt.ylabel("Total Medals")
plt.xticks(rotation = 45, ha="right")
plt.show()
top10SailingGold = dataMedal[(dataMedal.Sport == "Sailing") & (dataMedal.Medal == "Gold")].NOC.value_counts()[:11]
plt.figure(figsize=(20,6))
sns.barplot(x=top10SailingGold.index,y=top10SailingGold.values,palette = "RdPu_r")
plt.title("Top 10 Countries Have Most Gold Medals in Sailing")
plt.ylabel("Total Medals")
plt.xticks(rotation = 45, ha="right")
plt.show()
top10Rowing = dataMedal[dataMedal.Sport == "Rowing"].Team.value_counts()[:11]
plt.figure(figsize=(20,6))
sns.barplot(x=top10Rowing.index,y=top10Rowing.values,palette = "deep")
plt.title("Top 10 Countries Have Most Medals in Rowing")
plt.ylabel("Total Medals")
plt.xticks(rotation = 45, ha="right")
plt.show()
top10RowingGold = dataMedal[(dataMedal.Sport == "Rowing") & (dataMedal.Medal == "Gold")].Team.value_counts()[:11]
plt.figure(figsize=(20,6))
sns.barplot(x=top10RowingGold.index,y=top10RowingGold.values,palette = "muted")
plt.title("Top 10 Countries Have Most Gold Medals in Rowing")
plt.ylabel("Total Gold Medals")
plt.xticks(rotation = 45, ha="right")
plt.show()
top10Fencing = dataMedal[dataMedal.Sport == "Fencing"].NOC.value_counts()[:11]
plt.figure(figsize=(20,6))
sns.barplot(x=top10Fencing.index,y=top10Fencing.values,palette = "dark")
plt.title("Top 10 Countries Have Most Medals in Fencing")
plt.ylabel("Total Medals")
plt.xticks(rotation = 45, ha="right")
plt.show()
top10FencingGold = dataMedal[(dataMedal.Sport == "Fencing") & (dataMedal.Medal == "Gold")].NOC.value_counts()[:11]
plt.figure(figsize=(20,6))
sns.barplot(x=top10FencingGold.index,y=top10FencingGold.values,palette = "bright")
plt.title("Top 10 Countries Have Most Gold Medals in Fencing")
plt.ylabel("Total Gold Medals")
plt.xticks(rotation = 45, ha="right")
plt.show()
top10Boxing = dataMedal[dataMedal.Sport == "Boxing"].Team.value_counts()[:11]
plt.figure(figsize=(20,6))
sns.barplot(x=top10Boxing.index,y=top10Boxing.values,palette = "RdPu_r")
plt.title("Top 10 Countries Have Most Medals in Boxing")
plt.ylabel("Total Medals")
plt.xticks(rotation = 45, ha="right")
plt.show()
top10BoxingGold = dataMedal[(dataMedal.Sport == "Boxing") & (dataMedal.Medal == "Gold")].Team.value_counts()[:11]
plt.figure(figsize=(20,6))
sns.barplot(x=top10BoxingGold.index,y=top10BoxingGold.values,palette = "Reds_r")
plt.title("Top 10 Countries Have Most Gold Medals in Boxing")
plt.ylabel("Total Gold Medals")
plt.xticks(rotation = 45, ha="right")
plt.show()
top10Weightlifting = dataMedal[dataMedal.Sport == "Weightlifting"].Team.value_counts()[:11]
plt.figure(figsize=(20,6))
sns.barplot(x=top10Weightlifting.index,y=top10Weightlifting.values,palette = "seismic")
plt.title("Top 10 Countries Have Most Medals in Weightlifting")
plt.ylabel("Total Medals")
plt.xticks(rotation = 45, ha="right")
plt.show()
top10WeightliftingGold = dataMedal[(dataMedal.Sport == "Weightlifting") & (dataMedal.Medal == "Gold")].Team.value_counts()[:11]
plt.figure(figsize=(20,6))
sns.barplot(x=top10WeightliftingGold.index,y=top10WeightliftingGold.values,palette = "coolwarm_r")
plt.title("Top 10 Countries Have Most Gold Medals in Weightlifting")
plt.ylabel("Total Gold Medals")
plt.xticks(rotation = 45, ha="right")
plt.show()
top10Archery = dataMedal[dataMedal.Sport == "Archery"].NOC.value_counts()[:11]
plt.figure(figsize=(20,6))
sns.barplot(x=top10Archery.index,y=top10Archery.values,palette = "viridis")
plt.title("Top 10 Countries Have Most Medals in Archery")
plt.ylabel("Total Medals")
plt.xticks(rotation = 45, ha="right")
plt.show()
top10ArcheryGold = dataMedal[(dataMedal.Sport == "Archery") & (dataMedal.Medal == "Gold")].NOC.value_counts()[:11]
plt.figure(figsize=(20,6))
sns.barplot(x=top10ArcheryGold.index,y=top10ArcheryGold.values,palette = "gnuplot_r")
plt.title("Top 10 Countries Have Most Gold Medals in Archery")
plt.ylabel("Total Gold Medals")
plt.xticks(rotation = 45, ha="right")
plt.show()
data = [go.Box(x=list(df.Age.values))]
iplot(data)

ageAndMedal = dataMedal.iloc[:,[3,14]].Age.value_counts()
ageAndMedalList = sorted(zip(ageAndMedal.index,ageAndMedal.values))
agesList, medalList = zip(*ageAndMedalList)
agesList, medalList = list(agesList), list(medalList)
plt.figure(figsize=(20,10))
plt.scatter(agesList,medalList)
plt.xlabel("Age")
plt.ylabel("Medals")
plt.show()
dataMedal["CountMedal"] = 1
age0_17Medals = dataMedal[dataMedal.Age < 18].groupby("Medal").sum()
age18_24Medals = dataMedal[(dataMedal.Age > 17) & (dataMedal.Age < 25)].groupby("Medal").sum()
age25_34Medals = dataMedal[(dataMedal.Age > 24) & (dataMedal.Age < 35)].groupby("Medal").sum()
age35Medals = dataMedal[dataMedal.Age > 34].groupby("Medal").sum()
x = ["0-17","18-24","25-34","35+"]
y1 = [age0_17Medals.CountMedal[1], age18_24Medals.CountMedal[1], age25_34Medals.CountMedal[1],age35Medals.CountMedal[1]]
y2 = [age0_17Medals.CountMedal[2], age18_24Medals.CountMedal[2], age25_34Medals.CountMedal[2],age35Medals.CountMedal[2]]
y3 = [age0_17Medals.CountMedal[0], age18_24Medals.CountMedal[0], age25_34Medals.CountMedal[0],age35Medals.CountMedal[0]]

agesRelationshipMedal = pd.DataFrame([y1,y2,y3],columns=x, index = ["Gold","Silver","Bronze"])
agesRelationshipMedal
trace1 = go.Bar(
    x=x,
    y=y1,
    name = "Gold",
    text=y1,
    textposition = 'auto',
    marker=dict(
        color='#D4D621',
        line=dict(
            color='#000000',
            width=1.5),
        ),
    opacity=0.6
)

trace2 = go.Bar(
    x=x,
    y=y2,
    name = "Silver",
    text=y2,
    textposition = 'auto',
    marker=dict(
        color='#6B6262',
        line=dict(
            color='#000000',
            width=1.5),
        ),
    opacity=0.6
)
trace3 = go.Bar(
    x=x,
    y=y3,
    name = "Bronze",
    text=y3,
    textposition = 'auto',
    marker=dict(
        color='#CF5015',
        line=dict(
            color='#000000',
            width=1.5),
        ),
    opacity=0.6
)

data = [trace1,trace2,trace3]

iplot(data)
dfWomen = dataMedal[dataMedal.Sex == "F"]
mostMedalCountryWomen = dfWomen.groupby("NOC").sum().iloc[:,5].sort_values(ascending=False)[:21]
plt.figure(figsize=(20,12))
plt.xlabel("Medals")
plt.ylabel("Country")
sns.barplot(x=mostMedalCountryWomen.values, y=mostMedalCountryWomen.index, palette="tab20b")
plt.show()
for i in sorted(list(dataMedal.Sport.unique())):
    print(i)
    print("----------------------------")
    try:
        print("[Male]",dataMedal[(dataMedal.Sport == i) & (dataMedal.Sex == "M")].groupby("Name").sum().CountMedal.sort_values(ascending=False).index[0],":",dataMedal[(dataMedal.Sport == i) & (dataMedal.Sex == "M")].groupby("Name").sum().CountMedal.sort_values(ascending=False).values[0],"Medals"," - ",dataMedal[dataMedal.Name == dataMedal[(dataMedal.Sport == i) & (dataMedal.Sex == "M")].groupby("Name").sum().CountMedal.sort_values(ascending=False).index[0]].NOC.values[0])
        print("[Female]",dataMedal[(dataMedal.Sport == i) & (dataMedal.Sex == "F")].groupby("Name").sum().CountMedal.sort_values(ascending=False).index[0],":",dataMedal[(dataMedal.Sport == i) & (dataMedal.Sex == "F")].groupby("Name").sum().CountMedal.sort_values(ascending=False).values[0],"Medals"," - ",dataMedal[dataMedal.Name == dataMedal[(dataMedal.Sport == i) & (dataMedal.Sex == "F")].groupby("Name").sum().CountMedal.sort_values(ascending=False).index[0]].NOC.values[0])
    except IndexError:
        pass
    print("*************************************************************************************************")