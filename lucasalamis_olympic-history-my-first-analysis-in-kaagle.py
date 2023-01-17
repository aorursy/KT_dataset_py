import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.plotly as py

import plotly.graph_objs as go 

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True) 

%matplotlib inline
df = pd.read_csv('../input/athlete-events/athlete_events.csv')

df.head()
top_5_gold = df[df['Medal'] == 'Gold']['Team'].value_counts().head(5)
top_fg = pd.DataFrame(top_5_gold)
top_fg.reset_index(inplace=True)
top_fg
top_fg.rename(columns={'index':'Country'}, inplace=True)

plt.figure(figsize=(12,8))

sns.barplot(x='Country', y='Team', data=top_fg)
df['Age'].mean()
df[df['Medal'] == 'Gold']['Age'].mean()


df[df['Medal'] == 'Silver']['Age'].mean()
df[df['Medal'] == 'Bronze']['Age'].mean()
df['Age'].min()
df['Age'].max()
df.describe()
cat_by_year = df.groupby('Year')['Sport'].nunique()

cat_by_year = pd.DataFrame(cat_by_year)
cat_by_year.reset_index(inplace=True)
cat_by_year.head()
plt.figure(figsize=(20,8))

sns.barplot(x='Year', y='Sport', data=cat_by_year, palette='inferno')
df.groupby('NOC')['Medal'].count().head()
med_by_country = df.groupby('NOC')['Medal'].count()

med_by_country = pd.DataFrame(med_by_country)

med_by_country.reset_index(inplace=True)

med_by_country.head()

med_by_country = med_by_country.sort_values(by='Medal',ascending=False).head()

med_by_country
df.groupby('Season')['City'].nunique()
df['Event'].nunique()

most_played = df['Event'].value_counts().head()

most_played = pd.DataFrame(most_played)

most_played.reset_index(inplace=True)

most_played.rename(columns={'index':'Sport'}, inplace=True)



most_played

plt.figure(figsize=(16,8))

sns.barplot(x='Sport',y='Event', data=most_played)
df.corr()

most_act = df['NOC'].value_counts()
most_act = pd.DataFrame(most_act)
most_act.reset_index(inplace=True)
most_act.rename(columns={'index':'Country'}, inplace=True)
most_act.head()

data = dict(

        type = 'choropleth',

        locations = most_act['Country'],

        z = most_act['NOC'],

        text = most_act['Country'],

        colorbar = {'title' : 'Most active countries in Olympic Games'},

      ) 

layout = dict(

    title = 'Most active countries in Olympic Games',

    geo = dict(

        showframe = True

    )

)

choromap = go.Figure(data = [data],layout = layout)

iplot(choromap)

biggest_winners = df.groupby('Name')['Medal'].value_counts()

biggest_winners = pd.DataFrame(biggest_winners)

biggest_winners.sort_values(by='Medal', ascending=False).head()

df['Gold'] = pd.get_dummies(df['Medal']=='Gold', drop_first=True)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(df['NOC'])
df['country_cod'] = le.transform(df['NOC'])
le.fit(df['Event'])
le.transform(df['Event'])
df['event_cod'] = le.transform(df['Event'])
df.head()
from sklearn.model_selection import train_test_split
df['Age'].fillna(df['Age'].mean(), inplace=True)
x = df[['Age', 'Year', 'country_cod', 'event_cod']]

y = df['Gold']
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=101, test_size=0.30)
from sklearn.ensemble import RandomForestClassifier
rcf = RandomForestClassifier()
rcf.fit(x_train, y_train)
pred = rcf.predict(x_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, pred))