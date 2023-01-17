import numpy as np 

import pandas as pd 

import csv

from datetime import datetime

import re

import math

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px



%matplotlib inline
data = pd.read_csv("../input/nba2k20-player-dataset/nba2k20-full.csv")
data.head()
def clean_data(data):

    data['salary'] = data['salary'].apply(lambda x: int(x[1:]))

    data['jersey'] = data['jersey'].apply(lambda x: int(x[1:]))

    data['b_day'] = data['b_day'].apply(lambda x: datetime.strptime(x, '%m/%d/%y').date())

    data['height'] = data['height'].apply(lambda x: float(x[2+x.find('/'):]))

    data['weight'] = data['weight'].apply(lambda x: float(x[2+x.find('/'):-4]))

    data['draft_round'] = data['draft_round'].apply(lambda x: int(x) if len(x) == 1 else 0)

    data['draft_peak'] = data['draft_peak'].apply(lambda x: int(x) if 1<=len(x)<=2 else 0)

    data['college'] = data['college'].fillna('no education')

    data['team'] = data['team'].fillna('no team')
clean_data(data)
#find age of each player



def age_(birthday):

    today = datetime.strptime(datetime.today().strftime('%Y-%m-%d'), '%Y-%m-%d').date()

    age = today.year - birthday.year

    return int(age)



data['age'] = data['b_day'].apply(lambda x: age_(x))

data
plt.figure(figsize=(20, 10))

sns.heatmap(data.corr(),cmap='coolwarm',annot=True)
fig = px.scatter(

    data,

    x = 'salary',

    y = 'rating',

    color = 'position',

    title="Rating and salary correlation", 

    width=800,

    height=500,

    trendline="ols"

)



fig.show()
fig = px.histogram(data, x="salary", marginal="violin", width=900,height=500)

fig.show()
ds = data['rating'].value_counts().reset_index()

ds.columns = ['rating', 'count']

fig = px.bar(

    ds, 

    x='rating', 

    y="count", 

    orientation='v', 

    title='Players and their rating', 

    width=900,

    color='rating',

    height=500

)

fig.show()
data.loc[data['position'] == 'F-G', 'position'] = 'G-F'

data.loc[data['position'] == 'F-C', 'position'] = 'C-F'

ds = data['position'].value_counts().reset_index()

ds.columns = ['position', 'count of players']

fig = px.bar(

    ds, 

    x='position', 

    y="count of players", 

    orientation='v', 

    title="Players on positions", 

    width=800,

    height=500

)

fig.show()
fig = px.bar(

    data, 

    x='age', 

    y="rating", 

    orientation='v', 

    title="Age and rating ", 

    width=800,

    height=500,

    color = 'rating'

)

fig.show()
ds = data['team'].value_counts().reset_index()

ds.columns = ['team', 'count']

ds['mean rating'] = ds['team'].apply(lambda x: data[data['team'] == x]['rating'].mean())

ds.sort_values('mean rating',inplace=True, ascending = True)

ax = px.bar(ds, x="mean rating", y="team", color = 'mean rating').update_xaxes(categoryorder = 'total ascending')

ax
def index_ketle(height, weight):

    ik = float(weight)/((float(height))**2)

    if float(ik)<18.5:

        return 'underweight'

    elif 18.5<=float(ik)<=24.9:

        return 'normal body weight'

    else:

        return 'overweight'



ds = data['height'].reset_index()

ds['weight'] = data['weight']

del ds['index']



ds['Index Ketle'] = 0

for i in range(len(ds)):

    ds['Index Ketle'][i] = index_ketle(ds['height'][i], ds['weight'][i])



ds['Index Ketle'].unique()

ds



plt.figure(figsize=(15, 6))

b = sns.barplot(data=ds, x='height', y='weight', hue='Index Ketle', palette="Blues_d")

b.set_title("Index of body mass", fontsize=20)

b.set_xlabel("Height",fontsize=15)

b.set_ylabel("Weight",fontsize=15)

b
ds = data['college'].value_counts().reset_index()

ds.columns = ['college', 'count']

ds['mean reating'] = ds['college'].apply(lambda x: data[data['college'] == x]['rating'].mean())

ds.sort_values("mean reating", axis = 0, ascending = False, 

                 inplace = True, na_position ='last') 



plt.figure(figsize=(15, 8))

b = sns.barplot(data = ds.iloc[:10], x = 'college', y = 'mean reating', palette="Set3")

b.set_title("10 collages with top rated graduates", fontsize=20)

b.set_xlabel("College",fontsize=15)

b.set_ylabel("Mean reating",fontsize=15)
ds = data['height'].value_counts().reset_index()

ds.columns = ['height', 'count']

ds.sort_values('height', ascending= False, inplace=True)



plt.figure(figsize=(15, 8))

b = sns.barplot(data = ds, x = ds['height'][:10], y = ds['count'][:10], palette="Set3")

b.set_title("How many players with great height", fontsize=20)

b.set_xlabel("Height",fontsize=15)

b.set_ylabel("Count",fontsize=15)
ds = data['age'].value_counts().reset_index()

ds.columns = ['age', 'count']

ds['salary'] = ds['age'].apply(lambda x: data[data['age'] == x]['salary'].mean())

ds.sort_values('age', ascending= False, inplace=True)



plt.figure(figsize=(15, 8))

b = sns.barplot(data = ds, x = ds['age'], y = ds['salary'])

b.set_title("Age and mean salary", fontsize=20)

b.set_xlabel("Age",fontsize=15)

b.set_ylabel("Mean salary",fontsize=15)
ds = data[data['college'] == 'no education']

plt.figure(figsize=(15, 8))

b = sns.barplot(data=ds, x='rating', y='salary')

b.set_title("Rating and salary of players with no education", fontsize=20)

b.set_xlabel("Rating",fontsize=15)

b.set_ylabel("Salary",fontsize=15)

b
ds = data['draft_year'].value_counts().reset_index()

ds.columns = ['draft year', 'count']

ds.sort_values('draft year', ascending = True)



plt.figure(figsize=(15, 8))

b = sns.barplot(data=ds, x='draft year', y='count')

b.set_title("Amount of drafted players by the year", fontsize=20)

b.set_xlabel("Year",fontsize=15)

b.set_ylabel("Amount of players",fontsize=15)

b
ds = data['jersey'].value_counts().reset_index()

ds.columns = ['jersey', 'count']





fig, ax = plt.subplots(figsize=(12, 6))

ax.pie(ds['count'][:10], labels=ds['jersey'][:10], autopct='%1.1f%%',

        shadow=True, startangle=90)

ax.axis('equal')



plt.show()