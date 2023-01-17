# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #data visualization
import seaborn as sns #data visualization

#plotly library
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Load data from csv file
data=pd.read_csv('../input/athlete_events.csv')
#data includes how many rows and columns
data.shape
#column(feature) names in data
data.columns
#getting an overview of our data
data.info()
# checking for missing values
print("Are there missing values? {}".format(data.isnull().any().any()))
#missing value control in features
data.isnull().sum()
#Let's learn about the int values in our dataset.
data.describe() #include ID feature
#we don't need istaticsal summary for ID feature
data.iloc[:,1:].describe()
data.head()
#we found out how many teams in our data
print("Team Names in Dataset:")
print(data.Team.unique())

print("\nYears in Dataset:")
#we sorted the years  for a better look view.
print(np.sort(data.Year.unique()))

print("\nSport Types:")
print(data.Sport.unique())

#grouping by sex
data_gender=data['Sex'].value_counts()
data_gender.head()
average_height=data.Height.mean()
print("average height:",average_height)
#List Comprehension
data["Height_Level"]=["Short" if average_height>item else "Tall" for item in data.Height]

#pivot table grouping data by Height
pd.pivot_table(data,index="Height_Level",values="Height")

#List comprehension
average_age=data.Age.mean()
print("Average Age:",average_age)
data["Age_Level"]=["Old" if item>average_age else "Young" for item in data.Age]
data.loc[:10,["Age_Level","Name","Age"]]
data_gold=data[np.logical_and(data['Age']<18,data['Medal']=="Gold")]
data_gold.head(15)
#filtering pandas
#data[(data['Age']<18) & (data['Medal']=="Gold")]
#filtering pandas dataframe
data_team=data['Team']=="Turkey"
data[data_team].head(15)
data_first=data[np.logical_and(data['Team']=="Turkey",data['Medal']=="Gold")]
for index,value in data_first[0:1].iterrows():
    print(index,":",value)
#default function
def find(year,sport,Medal="Gold"):
    """
    parameter: Year,Sport type
    return: Athletes who receive a gold medal according to the sport type and the year.
    """
    data_find=data[(data['Year']==year) & (data['Sport']==sport) & (data['Medal']==Medal)]
    return data_find
find(2000,"Wrestling")
data_team=data[data.Team=="Turkey"]
data_turkey=data_team.loc[:,["ID","Year"]]

plt.figure(figsize=(12,16))
plt.subplot(211)

turkey = data_turkey.groupby("Year")["ID"].nunique().plot(kind = "bar",
                                                 color = sns.color_palette("husl"),
                                                 linewidth = 1)
plt.xticks(rotation = 60)
plt.grid(True,alpha=.3)
plt.show()
data_team=data[data.Team=="Turkey"]
data_turkey=data_team.loc[:,["ID","Year"]]
data_gold=data[np.logical_and(data['Team']=="Turkey",data['Medal']=="Gold")]

data1=data_turkey["Year"].value_counts(dropna=False).to_frame()
data2=data_gold["Year"].value_counts(dropna=False).to_frame()

data_percent=pd.concat([data1,data2],axis=1)
data_percent["year"]=data_percent.index
names=["participating","winner","year"]
data_percent.columns=names

data_percent.dropna(inplace=True)
data_percent.index=range(0,15)

def percent(x,y):
    result=round(x/y*2,2)
    return result
for item in data_percent:
    data_percent["percent"]=percent(data_percent["winner"],data_percent["participating"])

plt.figure(figsize=(12,16))
plt.subplot(211)    

ax = sns.barplot(x="year", y="percent", data=data_percent)
ax.set(ylabel="Percent")

plt.xticks(rotation = 60)
plt.grid(True,alpha=.3)
plt.show()

df = pd.DataFrame(data['Team'].value_counts())
df['country'] = df.index
df.columns = ['number', 'country']
df = df.reset_index().drop('index', axis=1)
data = [ dict(
        type = 'choropleth',
        locations = df['country'],
        locationmode = 'country names',
        z = df['number'],
        text = df['country'],
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '$',
            title = 'Olimpiyata Participants'),
      ) ]
layout = dict(
    title = 'Country of Participants',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)
fig = dict( data=data, layout=layout )
iplot( fig, validate=False, filename='world-map')
