import numpy as np

import pandas as pd

import plotly.graph_objs as go

import plotly.express as px

import matplotlib.pyplot as plt

from statsmodels.tsa.arima_model import ARIMA
df = pd.read_csv('/kaggle/input/us-police-shootings/shootings.csv')

df.head(10)
for col in df.columns:

    print(col, str(round(100* df[col].isnull().sum() / len(df), 2)) + '%')
data = df['manner_of_death'].value_counts().reset_index()

data.columns = ['manner_of_death', 'count']

fig = px.pie(

    data, 

    values='count', 

    names='manner_of_death', 

    title='Manner of death', 

    width=500, 

    height=500

)

fig.show()
data = df['armed'].value_counts().reset_index()

data.columns = ['armed', 'count']

data = data.sort_values('count')

fig = px.bar(

    data.tail(25), 

    x='count', 

    y='armed', 

    orientation='h', 

    title='Weapon', 

    width=800, 

    height=800

)

fig.show()
fig = px.histogram(

    df, 

    "age", 

    nbins=80, 

    title ='Age distribution', 

    width=800,

    height=500

)

fig.show()
fig = go.Figure(

    data=go.Violin(

        y=df['age'], 

        x0='Age'

    )

)

fig.show()
data = df['gender'].value_counts().reset_index()

data.columns = ['gender', 'count']



fig = px.pie(

    data, 

    values='count', 

    names='gender',  

    title='Gender distribution', 

    width=500, 

    height=500

)



fig.show()
data = df['race'].value_counts().reset_index()

data.columns = ['race', 'count']

data = data.sort_values('count')



fig = px.bar(

    data, 

    x='count', 

    y='race', 

    orientation='h', 

    title='Race distribution', 

    width=600,

    height=600

)



fig.show()
city = df.groupby('city')['name'].count().reset_index().sort_values('name', ascending=True).tail(50)



fig = px.bar(

    city, 

    x="name", 

    y="city", 

    orientation='h',

    title="Top 50 cities by deaths", 

    width=800, 

    height=900

)



fig.show()
df["date"] = pd.to_datetime(df["date"])

df["weekday"] = df["date"].dt.weekday

df['month'] = df['date'].dt.month

df['month_day'] = df['date'].dt.day

df['year'] = df['date'].dt.year
data = df.groupby(['weekday'])['name'].count().reset_index()

data.columns = ['weekday', 'count']



fig = px.bar(

    data, 

    x='weekday', 

    y='count',

    orientation='v', 

    title='Day of week', 

    width=600

)



fig.show()
data = df.groupby(['month'])['name'].count().reset_index()

data.columns = ['month', 'count']



fig = px.bar(

    data, 

    x='month', 

    y='count', 

    orientation='v', 

    title='Month of year', 

    width=800

)



fig.show()
data = df.groupby(['month_day'])['name'].count().reset_index()

data.columns = ['month_day', 'count']

fig = px.bar(

    data, 

    x='month_day', 

    y='count', 

    orientation='v', 

    title='Day of month', 

    width=800

)

fig.show()
data = df.groupby(['year'])['name'].count().reset_index()

data.columns = ['year', 'count']

fig = px.bar(

    data, 

    x='year', 

    y='count',

    orientation='v', 

    title='Year', 

    width=600

)

fig.show()
df['month_year'] = pd.to_datetime(df['date']).dt.to_period('M')

df.head()
data = df.groupby(['month_year'])['name'].count().reset_index()

data.columns = ['month_year', 'count']

data['month_year'] = data['month_year'].astype(str)

fig = px.bar(

    data, 

    x='month_year', 

    y='count', 

    orientation='v', 

    title='Deaths by months', 

    width=800

)

fig.show()
data = df.groupby(['flee'])['name'].count().reset_index()

data.columns = ['flee', 'count']

fig = px.bar(

    data, 

    x='flee', 

    y='count', 

    orientation='v', 

    title='Flee distribution', 

    width=600

)

fig.show()
data = df.groupby(['month_year'])['name'].count().reset_index()

data.columns = ['month_year', 'count']

data['month_year'] = data['month_year'].astype(str)

data = data.head(65)

fig = px.line(

    data, 

    x="month_year", 

    y="count", 

    title='Deaths month by month'

)

fig.show()
model = ARIMA(data['count'], order=(3,1,1))

model_fit = model.fit(disp=0)

print(model_fit.summary())
model_fit.plot_predict(dynamic=False)

plt.show()
preds, _, _ = model_fit.forecast(19, alpha=0.05)

preds = preds.tolist()

preds = [int(item) for item in preds]

months = ['2020-06', '2020-07', '2020-08', '2020-09', '2020-10', '2020-11', '2020-12', '2021-01', '2021-02', '2021-03', 

          '2021-04', '2021-05', '2021-06', '2021-07', '2021-08', '2021-09', '2021-10', '2021-11', '2021-12']



new_df = pd.DataFrame()

new_df['month_year'] = months

new_df['count'] = preds

data = pd.concat([data, new_df])
fig = px.line(

    data, 

    x="month_year", 

    y="count", 

    title='Deaths month by month with predictions'

)

fig.show()