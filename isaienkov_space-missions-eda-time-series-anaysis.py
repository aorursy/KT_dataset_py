import numpy as np

import pandas as pd

import plotly.express as px

from iso3166 import countries

from statsmodels.tsa.arima_model import ARIMA

import matplotlib

import matplotlib.pyplot as plt

from datetime import datetime, timedelta

from collections import OrderedDict

from statsmodels.tsa.seasonal import seasonal_decompose
df = pd.read_csv('/kaggle/input/all-space-missions-from-1957/Space_Corrected.csv')

df.columns = ['Unnamed: 0', 'Unnamed: 0.1', 'Company Name', 'Location', 'Datum', 'Detail', 'Status Rocket', 'Rocket', 'Status Mission']

df = df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)

df.head()
missed = pd.DataFrame()

missed['column'] = df.columns

percent = list()



for col in df.columns:

    percent.append(round(100* df[col].isnull().sum() / len(df), 2))



missed['percent'] = percent

missed = missed.sort_values('percent')

missed = missed[missed['percent']>0]



fig = px.bar(

    missed, 

    x='percent', 

    y="column", 

    orientation='h', 

    title='Missed values percent for every column (percent > 0)', 

    width=600,

    height=200 

)



fig.show()
ds = df['Company Name'].value_counts().reset_index()

ds.columns = ['company', 'number of starts']

ds = ds.sort_values(['number of starts'])



fig = px.bar(

    ds, 

    x='number of starts', 

    y="company", 

    orientation='h', 

    title='Number of launches by every company', 

    width=800,

    height=1000 

)



fig.show()
ds = df['Status Rocket'].value_counts().reset_index()

ds.columns = ['status', 'count']



fig = px.pie(

    ds, 

    values='count', 

    names="status", 

    title='Rocket status', 

    width=500, 

    height=500

)



fig.show()
ds = df['Status Mission'].value_counts().reset_index()

ds.columns = ['mission_status', 'count']



fig = px.bar(

    ds, 

    x='mission_status', 

    y="count", 

    orientation='v', 

    title='Mission Status distribution', 

    width=500,

    height=500

)



fig.show()
df['Rocket'] = df['Rocket'].fillna(0.0).str.replace(',', '')

df['Rocket'] = df['Rocket'].astype(np.float64).fillna(0.0)

df['Rocket'] = df['Rocket'] * 1000000
df.loc[df['Rocket']>4000000000, 'Rocket'] = 0.0



fig = px.histogram(

    df, 

    "Rocket", 

    nbins=50, 

    title='Rocket Value distribution', 

    width=700, 

    height=500

)



fig.show()
countries_dict = {

    'Russia' : 'Russian Federation',

    'New Mexico' : 'USA',

    "Yellow Sea": 'China',

    "Shahrud Missile Test Site": "Iran",

    "Pacific Missile Range Facility": 'USA',

    "Barents Sea": 'Russian Federation',

    "Gran Canaria": 'USA'

}



df['country'] = df['Location'].str.split(', ').str[-1].replace(countries_dict)
sun = df.groupby(['country', 'Company Name', 'Status Mission'])['Datum'].count().reset_index()

sun.columns = ['country', 'company', 'status', 'count']



fig = px.sunburst(

    sun, 

    path=[

        'country', 

        'company', 

        'status'

    ], 

    values='count', 

    title='Sunburst chart for all countries',

    width=600,

    height=600

)



fig.show()
country_dict = {}

for c in countries:

    country_dict[c.name] = c.alpha3

    

df['alpha3'] = df['country']

df = df.replace({"alpha3": country_dict})

df.loc[df['country'] == "North Korea", 'alpha3'] = "PRK"

df.loc[df['country'] == "South Korea", 'alpha3'] = "KOR"

df
def plot_map(dataframe, target_column, title, width=800, height=600):

    mapdf = dataframe.groupby(['country', 'alpha3'])[target_column].count().reset_index()

    fig = px.choropleth(

        mapdf, 

        locations="alpha3", 

        hover_name="country", 

        color=target_column, 

        projection="natural earth", 

        width=width, 

        height=height, 

        title=title

    )

    fig.show()
plot_map(df, 'Status Mission', 'Number of starts per country')
fail_df = df[df['Status Mission'] == 'Failure']

plot_map(fail_df, 'Status Mission', 'Number of Fails per country')
data = df.groupby(['Company Name'])['Rocket'].sum().reset_index()

data = data[data['Rocket'] > 0]

data.columns = ['company', 'money']



fig = px.bar(

    data, 

    x='company', 

    y="money", 

    orientation='v', 

    title='Total money spent on missions', 

    width=800,

    height=600

)



fig.show()
money = df.groupby(['Company Name'])['Rocket'].sum()

starts = df['Company Name'].value_counts().reset_index()

starts.columns = ['Company Name', 'count']



av_money_df = pd.merge(money, starts, on='Company Name')

av_money_df['avg'] = av_money_df['Rocket'] / av_money_df['count']

av_money_df = av_money_df[av_money_df['avg']>0]

av_money_df = av_money_df.reset_index()



fig = px.bar(

    av_money_df, 

    x='Company Name', 

    y="avg", 

    orientation='v', 

    title='Average money per one launch', 

    width=800,

    height=600

)



fig.show()
df['date'] = pd.to_datetime(df['Datum'])

df['year'] = df['date'].apply(lambda datetime: datetime.year)

df['month'] = df['date'].apply(lambda datetime: datetime.month)

df['weekday'] = df['date'].apply(lambda datetime: datetime.weekday())
ds = df['year'].value_counts().reset_index()

ds.columns = ['year', 'count']



fig = px.bar(

    ds, 

    x='year', 

    y="count", 

    orientation='v', 

    title='Missions number by year', 

    width=800

)



fig.show()
ds = df['month'].value_counts().reset_index()

ds.columns = ['month', 'count']



fig = px.bar(

    ds, 

    x='month',

    y="count", 

    orientation='v', 

    title='Missions number by month', 

    width=800

)



fig.show()
ds = df['weekday'].value_counts().reset_index()

ds.columns = ['weekday', 'count']



fig = px.bar(

    ds, 

    x='weekday', 

    y="count", 

    orientation='v',

    title='Missions number by weekday', 

    width=800

)



fig.show()
res = list()

for group in df.groupby(['Company Name']):

    res.append(group[1][['Company Name', 'year']].head(1))



data = pd.concat(res)

data = data.sort_values('year')

data['year'] = 2020 - data['year']



fig = px.bar(

    data, 

    x="year", 

    y="Company Name", 

    orientation='h', 

    title='Years from last start',

    width=900,

    height=1000

)



fig.show()
money = df[df['Rocket']>0]

money = money.groupby(['year'])['Rocket'].mean().reset_index()



fig = px.line(

    money, 

    x="year", 

    y="Rocket",

    title='Average money spent by year'

)



fig.show()
ds = df.groupby(['Company Name'])['year'].nunique().reset_index()

ds.columns = ['company', 'count']



fig = px.bar(

    ds, 

    x="company", 

    y="count", 

    title='Most experienced companies (years of launches)'

)



fig.show()
data = df.groupby(['Company Name', 'year'])['Status Mission'].count().reset_index()

data.columns = ['company', 'year', 'starts']

top5 = data.groupby(['company'])['starts'].sum().reset_index().sort_values('starts', ascending=False).head(5)['company'].tolist()
data = data[data['company'].isin(top5)]



fig = px.line(

    data, 

    x="year", 

    y="starts", 

    title='Dynamic of top 5 companies by number of starts', 

    color='company'

)



fig.show()
data = df.groupby(['Company Name', 'year'])['Status Mission'].count().reset_index()

data.columns = ['company', 'year', 'starts']

data = data[data['year']==2020]
fig = px.bar(

    data, 

    x="company", 

    y="starts", 

    title='Number of starts for 2020', 

    width=800

)



fig.show()
data = df[df['Status Mission']=='Failure']

data = data.groupby(['Company Name', 'year'])['Status Mission'].count().reset_index()

data.columns = ['company', 'year', 'starts']

data = data[data['year']==2020]



fig = px.bar(

    data, 

    x="company", 

    y="starts", 

    title='Failures in 2020', 

    width=600

)



fig.show()
data = df[df['Company Name'] == 'CASC']

data = data.groupby(['year'])['Company Name'].count().reset_index()

data = data[data['year'] < 2020]

data.columns = ['year', 'launches']



fig = px.line(

    data, 

    x="year", 

    y="launches", 

    title='Launches per year for CASC'

)



fig.show()
model = ARIMA(data['launches'], order=(2,2,1))

model_fit = model.fit(disp=0)

model_fit.summary()
model_fit.plot_predict(dynamic=False)

plt.show()
preds, _, _ = model_fit.forecast(6, alpha=0.05)



preds = preds.tolist()

preds = [int(item) for item in preds]

months = ['2020', '2021', '2022', '2023', '2024', '2025']



new_df = pd.DataFrame()

new_df['year'] = months

new_df['launches'] = preds

data = pd.concat([data, new_df])



fig = px.line(

    data, 

    x="year", 

    y="launches", 

    title='Launches per year prediction for CASC'

)



fig.show()
cold = df[df['year'] <= 1991]

cold['country'].unique()

cold.loc[cold['country'] == 'Kazakhstan', 'country'] = 'USSR'

cold.loc[cold['country'] == 'Russian Federation', 'country'] = 'USSR'

cold = cold[(cold['country'] == 'USSR') | (cold['country'] == 'USA')]
ds = cold['country'].value_counts().reset_index()

ds.columns = ['contry', 'count']



fig = px.pie(

    ds, 

    names='contry', 

    values="count", 

    title='Number of launches', 

    width=500

)



fig.show()
ds = cold.groupby(['year', 'country'])['alpha3'].count().reset_index()

ds.columns = ['year', 'country', 'launches']



fig = px.bar(

    ds, 

    x="year", 

    y="launches", 

    color='country', 

    title='USA vs USSR: launches year by year',

    width=800

)



fig.show()
ds = cold.groupby(['year', 'country'])['Company Name'].nunique().reset_index()

ds.columns = ['year', 'country', 'companies']



fig = px.bar(

    ds, 

    x="year", 

    y="companies", 

    color='country', 

    title='USA vs USSR: number of companies year by year',

    width=800

)



fig.show()
ds = cold[cold['Status Mission'] == 'Failure']

ds = ds.groupby(['year', 'country'])['alpha3'].count().reset_index()

ds.columns = ['year', 'country', 'failures']



fig = px.bar(

    ds, 

    x="year", 

    y="failures", 

    color='country', 

    title='USA vs USSR: failures year by year', 

    width=800

)



fig.show()
ds = df.groupby(['year', 'country'])['Status Mission'].count().reset_index().sort_values(['year', 'Status Mission'], ascending=False)

ds = pd.concat([group[1].head(1) for group in ds.groupby(['year'])])

ds.columns = ['year', 'country', 'launches']



fig = px.bar(

    ds, 

    x="year", 

    y="launches", 

    color='country', 

    title='Leaders by launches for every year (countries)'

)



fig.show()
ds = df[df['Status Mission']=='Success']

ds = ds.groupby(['year', 'country'])['Status Mission'].count().reset_index().sort_values(['year', 'Status Mission'], ascending=False)

ds = pd.concat([group[1].head(1) for group in ds.groupby(['year'])])

ds.columns = ['year', 'country', 'launches']



fig = px.bar(

    ds, 

    x="year", 

    y="launches", 

    color='country', 

    title='Leaders by success launches for every year (countries)'

)



fig.show()
ds = df.groupby(['year', 'Company Name'])['Status Mission'].count().reset_index().sort_values(['year', 'Status Mission'], ascending=False)

ds = pd.concat([group[1].head(1) for group in ds.groupby(['year'])])

ds.columns = ['year', 'company', 'launches']



fig = px.bar(

    ds, 

    x="year", 

    y="launches", 

    color='company', 

    title='Leaders by launches for every year (companies)'

)



fig.show()
ds = df[df['Status Mission']=='Success']

ds = ds.groupby(['year', 'Company Name'])['Status Mission'].count().reset_index().sort_values(['year', 'Status Mission'], ascending=False)

ds = pd.concat([group[1].head(1) for group in ds.groupby(['year'])])

ds.columns = ['year', 'company', 'launches']



fig = px.bar(

    ds, 

    x="year", 

    y="launches", 

    color='company', 

    title='Leaders by success launches for every year (companies)'

)



fig.show()
df['month_year'] = df['year'].astype(str) + '-' + df['month'].astype(str)

df['month_year'] = pd.to_datetime(df['month_year']).dt.to_period('M')

ds = df.groupby(['month_year'])['alpha3'].count().reset_index()

ds.columns = ['month_year', 'count']

ds['month_year'] = ds['month_year'].astype(str)



fig = px.line(

    ds, 

    x='month_year', 

    y='count', 

    orientation='v', 

    title='Launches by months' 

)



fig.show()
dates = ['1957-10-01', '2020-08-02']

start, end = [datetime.strptime(_, "%Y-%m-%d") for _ in dates]

dd = pd.DataFrame(list(OrderedDict(((start + timedelta(_)).strftime(r"%Y-%m"), None) for _ in range((end - start).days)).keys()), columns=['date'])

dd['date'] = pd.to_datetime(dd['date'])

ds['month_year'] = pd.to_datetime(ds['month_year'])

res = pd.merge(ds, dd, how='outer', left_on='month_year', right_on='date')

res = res.sort_values('date')[['date', 'count']]

res = res.fillna(0).set_index('date')
result = seasonal_decompose(res, model='additive', freq=12)

fig = result.plot()

matplotlib.rcParams['figure.figsize'] = [20, 15]



plt.show()
ts = (result.trend + result.seasonal).reset_index()

ts.columns = ['date', 'count']

ts['origin'] = 'cleaned'

dres = res.reset_index()

dres['origin'] = 'original'

data = pd.concat([dres, ts])
fig = px.line(

    data, 

    x='date', 

    y='count', 

    color='origin', 

    orientation='v', 

    title='Original and cleaned time series', 

    width=800

)



fig.show()
model = ARIMA(ds['count'], order=(10,1,2))

model_fit = model.fit()
model_fit.plot_predict(dynamic=False)



plt.show()
preds, _, _ = model_fit.forecast(16)

preds = preds.tolist()

preds = [int(item) for item in preds]

months = ['2020-09-01', '2020-10-01', '2020-11-01', '2020-12-01', '2021-01-01', '2021-02-01', '2021-03-01', '2021-04-01', 

          '2021-05-01', '2021-06-01', '2021-07-01', '2021-08-01', '2021-09-01', '2021-10-01', '2021-11-01', '2021-12-01']



new_df = pd.DataFrame()

new_df['month_year'] = months

new_df['count'] = preds

data = pd.concat([ds, new_df])



fig = px.line(

    data, 

    x="month_year", 

    y="count", 

    title='Launches per month prediction'

)



fig.show()