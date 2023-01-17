from IPython.display import Image

Image("../input/nifty51/9.PNG")
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

%matplotlib inline

from pandas_profiling import ProfileReport



data = pd.read_csv('../input/nifty-indices-dataset/NIFTY 50.csv')

data['Date'] = pd.to_datetime(data['Date'])

data['year'] = data['Date'].dt.year

data['month'] = data['Date'].dt.month

data['day'] = data['Date'].dt.day
report = ProfileReport(data)
report
plt.figure(figsize=(10,7))

plt.plot(data['Date'],data['Close'])

plt.xlabel('Years')

plt.ylabel('Closing values')

plt.title('Closing values vs Years')

plt.show()
peaks = data.loc[:, ['year','High']]

peaks['max_high'] = peaks.groupby('year')['High'].transform('max')

peaks.drop('High', axis=1, inplace=True)

peaks = peaks.drop_duplicates()

peaks = peaks.sort_values('max_high', ascending=False)

peaks = peaks.head()



fig = plt.figure(figsize=(10,7))

plt.pie(peaks['max_high'], labels=peaks['year'], autopct='%1.1f%%', shadow=True)

centre_circle = plt.Circle((0,0),0.45,color='black', fc='white',linewidth=1.25)

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

plt.axis('equal')

plt.show()
sns.kdeplot(data=data['High'], shade=True)

plt.title('Distribution of highest values')

plt.show()
sns.kdeplot(data=data['Volume'], shade=True)

plt.title('Transaction volume')

plt.show()
top_5_genres = [1,3,5,7,9,11]

perc = data.loc[:,["year","month",'Volume']]

perc['new_volume'] = perc.groupby([perc.month,perc.year])['Volume'].transform('mean')

perc.drop('Volume', axis=1, inplace=True)

perc = perc[perc.year<2020]

perc = perc.drop_duplicates()

perc = perc.loc[perc['month'].isin(top_5_genres)]

perc = perc.sort_values("year")



fig=px.bar(perc,x='month', y="new_volume", animation_frame="year", 

           animation_group="month", color="month", hover_name="month", range_y=[perc['new_volume'].min(), perc['new_volume'].max()])

fig.update_layout(showlegend=False)

fig.show()
sns.scatterplot(data=data, x='Volume', y='Close')

plt.title('Relation of volume to stock prices')

plt.show()
sns.kdeplot(data=data['Turnover'], shade=True)

plt.title('Turnover Distribution')

plt.show()
sns.scatterplot(data=data, x='Turnover', y='Volume')

plt.title('Relation of Turnover to Volume')

plt.show()
turn = data.loc[:,['year','month','Turnover']]

turn['monthly_turnover'] = turn.groupby([turn.year,turn.month])['Turnover'].transform('mean')

turn.drop('Turnover', axis=1, inplace=True)

turn = turn.drop_duplicates()

fig = px.scatter(turn, x="month", y="monthly_turnover", animation_frame="year", animation_group="month", color="month", hover_name="month", size_max=1000 \

                , range_y=[turn['monthly_turnover'].min(), turn['monthly_turnover'].max()])

fig.update_traces(marker=dict(size=12,

                              line=dict(width=2,

                                        color='DarkSlateGrey')),

                  selector=dict(mode='markers'))

fig.show()
sns.kdeplot(data=data['P/E'], shade=True)

plt.title('P/E Distribution')

plt.show()
sns.scatterplot(data=data, x='P/E', y='Close')

plt.title('Relation of P/E to Close')

plt.show()
df = data.loc[:,['year','P/E']]

df['meanPE'] = df.groupby('year')['P/E'].transform('mean')

df.drop('P/E',axis=1, inplace=True)

df = df.drop_duplicates().sort_values('year')





plt.figure(figsize=(10,7))

plt.plot(df['year'],df['meanPE'])

plt.xlabel('Years')

plt.ylabel('P/E values')

plt.title('P/E values vs Years')

plt.show()
sns.kdeplot(data=data['P/B'], shade=True)

plt.title('P/B Distribution')

plt.show()
sns.scatterplot(data=data, x='P/B', y='Close', hue='year')

plt.title('Relation of P/B to Close')

plt.show()
df = data.loc[:,['year','P/B']]

df['meanPB'] = df.groupby('year')['P/B'].transform('mean')

df.drop('P/B',axis=1, inplace=True)

df = df.drop_duplicates().sort_values('year')





plt.figure(figsize=(10,7))

plt.plot(df['year'],df['meanPB'])

plt.xlabel('Years')

plt.ylabel('P/B values')

plt.title('P/B values vs Years')

plt.show()
sns.kdeplot(data=data['Div Yield'], shade=True)

plt.title('Div Yield Distribution')

plt.show()
sns.scatterplot(data=data, x='Div Yield', y='Close', hue='year')

plt.title('Relation of Div Yield to Close')

plt.show()
df = data.loc[:,['year','Div Yield']]

df['meandiv'] = df.groupby('year')['Div Yield'].transform('max')

df.drop('Div Yield',axis=1, inplace=True)

df = df.drop_duplicates().sort_values('year')





plt.figure(figsize=(10,7))

plt.plot(df['year'],df['meandiv'])

plt.xlabel('Years')

plt.ylabel('Div Yield values')

plt.title('Div Yield values vs Years')

plt.show()
df = data.loc[:,['year','P/B','P/E','Div Yield']]

df[['meanPE','meanPB','meandiv']] = df.groupby('year')[['P/B','P/E','Div Yield']].transform('max')

df.drop(['P/B','P/E','Div Yield'],axis=1, inplace=True)

df = df.drop_duplicates().sort_values('year')





plt.figure(figsize=(10,7))

plt.plot(df['year'],df['meandiv'], label='meandiv')

plt.plot(df['year'],df['meanPB'], label='meanPB')

plt.plot(df['year'],df['meanPE'], label='meanPE')

plt.xlabel('Years')

plt.legend()

plt.show()
df5 = data[(data['year']>=2005) & (data['year']<=2009)]
sns.lineplot(data=df5, x='Date', y='Close', hue='year')

plt.ylabel('Close points[a.u.]')

plt.show()
df8 = data[data['year']==2008]

sns.lineplot(data=df8, x='Date', y='Close', hue='month')

plt.ylabel('Close points[a.u.]')

plt.show()
df6 = data[data['year']==2006]

sns.lineplot(data=df6, x='Date', y='Close', hue='month')

plt.ylabel('Close points[a.u.]')

plt.show()