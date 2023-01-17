# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

plt.style.use('seaborn-talk')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from pylab import rcParams

rcParams['figure.figsize'] = 14,6
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score,mean_squared_error
data = pd.read_table('/kaggle/input/gapminder/gapminder.tsv')

data.head()
data.loc[data["year"]==2007]["continent"].value_counts()
def find(df,col,eq):

    return df.loc[df[col]==eq]
Africa = find(data,"continent","Africa")

Asia = find(data,"continent","Asia")

Europe = find(data,"continent","Europe")

Americas = find(data,"continent","Americas")

Oceania = find(data,"continent","Oceania")
Continent = data.groupby(['continent','year'])['pop'].sum().reset_index()

World = data.groupby(['year'])['pop'].sum().reset_index()

World.index = World["year"]

World = World.drop("year",axis=1)
World.style.background_gradient("Greens")
sns.swarmplot(data  = data.loc[data["year"]==2007], x="continent",y="lifeExp",color="0000")

sns.boxenplot(data  = data.loc[data["year"]==2007], x="continent",y="lifeExp")

plt.title("Life expectancy at birth")

plt.show()
sns.swarmplot(data  = data.loc[data["year"]==2007], x="continent",y="pop",color="0000")

sns.boxenplot(data  = data.loc[data["year"]==2007], x="continent",y="pop")

plt.title("World Population")

plt.show()
sns.swarmplot(data  = data.loc[data["year"]==2007], x="continent",y="gdpPercap",color="0000")

sns.boxenplot(data  = data.loc[data["year"]==2007], x="continent",y="gdpPercap")

plt.title("Per-capita GDP")

plt.show()
fig = px.scatter(x=data.loc[data['year']==2007]["gdpPercap"], y=data.loc[data['year']==2007]["lifeExp"], 

                 color=data.loc[data['year']==2007]["continent"],

                 size=data.loc[data['year']==2007]["pop"],title="x: gdpPercap y:lifeExp size:pop")

fig.show()
arr = [Africa,Asia,Europe,Americas,Oceania]



title = ["Population of Africa 2007",

        "Population of Asia 2007",

        "Population of Europe 2007",

        "Population of Americas 2007",

        "Population of Oceania 2007"]

for i in range(len(title)):

    if i==4:

        plt.figure(figsize=(10,1))

    else:

        plt.figure(figsize=(10,15))

    sns.barplot(y="country",x="pop",data=arr[i].loc[arr[i]["year"]==2007].sort_values("pop",ascending=False),

               palette='Set1')

    plt.title(title[i])

    plt.ylabel("Country")

    plt.xlabel("Population")

    plt.show()
arr = [Africa,Asia,Europe,Americas,Oceania]



title = ["Per-capita GDP of Africa 2007",

        "Per-capita GDP of Asia 2007",

        "Per-capita GDP of Europe 2007",

        "Per-capita GDP of Americas 2007",

        "Per-capita GDP of Oceania 2007"]

for i in range(len(title)):

    if i==4:

        plt.figure(figsize=(10,1))

    else:

        plt.figure(figsize=(10,15))

    sns.barplot(y="country",x="gdpPercap",data=arr[i].loc[arr[i]["year"]==2007].sort_values("gdpPercap",ascending=False),

               palette='Set2')

    plt.title(title[i])

    plt.ylabel("Country")

    plt.xlabel("Per-capita GDP")

    plt.show()
pop100M = data.loc[(data["pop"]>100000000) & (data["year"]==2007)].sort_values("pop",ascending=False)

pop_gdpPercap = data.loc[(data["gdpPercap"]>35000) & (data["year"]==2007)].sort_values("gdpPercap",ascending=False)
gdp = pd.DataFrame(pop_gdpPercap["gdpPercap"])

gdp.index = pop_gdpPercap["country"]

gdp2 = pd.DataFrame(data.loc[data["year"]==2007].sort_values("gdpPercap")[:15]["gdpPercap"])

gdp2.index = data.loc[data["year"]==2007].sort_values("gdpPercap")[:15]["country"]

gdp.columns =  ["Per-capita GDP"]

gdp2.columns =  ["Per-capita GDP"]
gdp.style.background_gradient("Oranges")
gdp2.style.background_gradient("Oranges")
world_continent = data.groupby(['continent'])['pop'].sum().reset_index()

world_continent.style.background_gradient("coolwarm")
fig = px.pie(pop100M, values='pop', names='country', title='Countries with a population of more than 100M')

fig.show()
fig = px.pie(world_continent, values='pop', names="continent" ,title='World Population 2007')

fig.show()
fig = px.bar(Continent, x='year', y='pop',color="continent",title="World population by years")

fig.show()
X = np.array(World.index).reshape(-1,1)

y = np.array(World["pop"]) 
MODEL = LinearRegression()

MODEL.fit(X,y)
pred = MODEL.predict(X)

plt.scatter(X,y,label="Actual",s=100)

plt.plot(X,pred,label="Regression",color='r')

plt.legend()

plt.grid(True)

plt.show()

print("New case for next 3 Yeras:",MODEL.predict([[2010],[2020],[2030]]))

print("MSE:",mean_squared_error(y,pred))

print("R2 :",r2_score(y,pred))