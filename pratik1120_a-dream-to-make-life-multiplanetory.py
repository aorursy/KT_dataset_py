import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.express as px

from wordcloud import WordCloud



import warnings

warnings.filterwarnings("ignore")



plt.rcParams['figure.figsize'] = 8, 5

plt.style.use("fivethirtyeight")

pd.options.plotting.backend = "plotly"



data = pd.read_csv('../input/all-space-missions-from-1957/Space_Corrected.csv')

data.drop(['Unnamed: 0','Unnamed: 0.1'], axis=1, inplace=True)
fig = data.nunique().reset_index().plot(kind='bar', x='index', y=0, color=0)

fig.update_layout(title='Unique Value Count Plot', xaxis_title='Variables', yaxis_title='Unique value count')

fig.show()
fig = data.isnull().sum().reset_index().plot(kind='bar', x='index', y=0)

fig.update_layout(title='Missing Value Plot', xaxis_title='Variables', yaxis_title='Missing value count')

fig.show()
fig = data['Status Rocket'].value_counts().reset_index().plot(kind='bar', x='index', y='Status Rocket', color='Status Rocket')

fig.update_layout(title='Status of all Rockets', xaxis_title='Rocket Status', yaxis_title='Count')

fig.show()
fig = data['Status Mission'].value_counts().reset_index().plot(kind='bar', x='index', y='Status Mission', color='Status Mission')

fig.update_layout(title='Status of all Missions', xaxis_title='Mission Status', yaxis_title='Count')

fig.show()
fig = data['Company Name'].value_counts().reset_index().head(11).plot(kind='bar',x='index',y='Company Name',color='Company Name')

fig.update_layout(title='Company with most Rocket launches',xaxis_title='Company', yaxis_title='No of Rocket launches')

fig.show()
best_com = data['Company Name'].value_counts().reset_index().head(11)['index'].tolist()

df = data[['Company Name','Status Mission']].copy().reset_index()

df['Count'] = df.groupby(['Status Mission','Company Name'])['index'].transform('count')

df = df[df['Company Name'].isin(best_com)]

df = df.drop('index',axis=1).drop_duplicates().reset_index(drop=True)

fig = df.plot(kind='bar', x='Company Name', y='Count', color='Status Mission', barmode='group')

fig.update_layout(title='Segmentation of launches by companies based on status',xaxis_title='Company',yaxis_title='Number of launches')

fig.show()
fig = data['Location'].value_counts().reset_index().head(11).plot(kind='bar',y='index',x='Location',color='Location')

fig.update_layout(title='Locations with most Rocket launches',yaxis_title='Location', xaxis_title='No of Rocket launches')

fig.show()
data['Date'] = pd.to_datetime(data['Datum'], utc=True)

data['year'] = data['Date'].apply(lambda x: x.year).astype('int')

fig = data['year'].value_counts().reset_index().plot(kind='bar',x='index',y='year',color='year')

fig.update_layout(title='Years with most Rocket launches',xaxis_title='year', yaxis_title='No of Rocket launches')

fig.show()
data['month'] = data['Date'].apply(lambda x: x.month).astype('int')

fig = data['month'].value_counts().reset_index().plot(kind='bar',x='index',y='month',color='month')

fig.update_layout(title='Months with most Rocket launches',xaxis_title='month', yaxis_title='No of Rocket launches')

fig.show()
count = data['Company Name'].value_counts().reset_index()

success = data.loc[data['Status Mission']=='Success','Company Name'].value_counts().reset_index()



merged = count.merge(success, on='index')

merged['success_rate'] = (merged['Company Name_y'] / merged['Company Name_x'])*100

merged = merged.head(15)



fig = merged.plot(kind='bar', x='index', y='success_rate', color='Company Name_x')

fig.update_layout(title='Success Rate of different companies', xaxis_title='Company Name', yaxis_title='Success Rate')

fig.show()
text = " ".join(review for review in data.Detail)

wordcloud = WordCloud(max_words=200, colormap='Set3',background_color="black").generate(text)

plt.figure(figsize=(15,10))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()