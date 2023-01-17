# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
color = sns.color_palette()
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
import plotly.tools as tls
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/Donors.csv")
df.head()
df.shape
df.isnull().sum()
df['Donor City'].fillna('Unknown',inplace=True) # Please see we're going to drop unknown when we visualize maximum cities
df['Donor State'].nunique()
cnt_srs = df.ix[df['Donor City']!='Unknown']['Donor City'].value_counts().head(20) # We are dropping cities with unknown values

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color="green",
        #colorscale = 'Blues',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Top 20 Cities with Highest Donors'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Donors")  


cnt_srs = df['Donor State'].value_counts().head(20) # We are dropping cities with unknown values

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color="green",
        #colorscale = 'Blues',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Top 20 States with Highest Donors'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Donors")  
df.head(1)
df['Donor Is Teacher'].unique()
df['Donor Is Teacher'] = df['Donor Is Teacher'].map({'No': 'Non-Teachers', 'Yes': 'Teachers'})
temp_series = df['Donor Is Teacher'].value_counts()
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(
    title="Donor's Categorical Distribution (Teachers or Non-Teachers)",
    width=600,
    height=600,
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Donor")
city_df=df.groupby('Donor City')
df2=pd.read_csv('../input/Donations.csv')
df2.head()
df2.dtypes
df2['Donation Received Date']=pd.to_datetime(df2['Donation Received Date'])
df2['Receiving_Date'] =pd.DatetimeIndex(df2['Donation Received Date']).normalize()
df2.head(1)
df2.shape
df2_dash=df2.tail(1000)
df_ts=df2_dash[['Donation Amount','Receiving_Date']]
df_ts.index=df_ts['Receiving_Date']
df_ts['Donation Amount'].plot(figsize=(15,6), color="green")
plt.xlabel('Year')
plt.ylabel('Donation Amount')
plt.title("Donation Amount Time-Series Visualization")
plt.show()
from fbprophet import Prophet
sns.set(font_scale=1) 
df_date_index = df2_dash[['Receiving_Date','Donation Amount']]
df_date_index = df_date_index.set_index('Receiving_Date')
df_prophet = df_date_index.copy()
df_prophet.reset_index(drop=False,inplace=True)
df_prophet.columns = ['ds','y']

m = Prophet()
m.fit(df_prophet)
future = m.make_future_dataframe(periods=30,freq='D')
forecast = m.predict(future)
fig = m.plot(forecast)
m.plot_components(forecast);
df3=pd.read_csv('../input/Resources.csv')
df3.head()
df3.isnull().sum()
df3.shape
df3=df3.dropna() #The dataset is huge, filling in with Imputers was taking a lot of time
df3.shape
df3.isnull().any()
df3.head(1)
df4=df3.groupby(['Resource Vendor Name'],as_index=False)['Resource Quantity','Resource Unit Price'].agg('sum')
df4.head(1)
df4['revenue']=df4['Resource Quantity']*df4['Resource Unit Price']
df4.head(1)
df4=df4.sort_values('Resource Quantity',ascending=False)
df4.shape
type_colors = ['#78C850',  # Grass
                    '#F08030',  # Fire
                    '#6890F0',  # Water
                    '#A8B820',  # Bug
                    '#A8A878',  # Normal
                    '#A040A0',  # Poison
                    '#F8D030',  # Electric
                    '#E0C068',  # Ground
                    '#EE99AC',  # Fairy
                    '#C03028',  # Fighting
                    '#F85888',  # Psychic
                    '#B8A038',  # Rock
                    '#705898',  # Ghost
                    '#98D8D8',  # Ice
                    '#7038F8',  # Dragon
                   ]

sns.set(rc={'figure.figsize':(11.7,8.27)}) 
ax= sns.barplot(x=df4['Resource Vendor Name'].head(10), y=df4['Resource Quantity'].head(10),palette = type_colors)
plt.xlabel('Name of the  Vendor',fontsize=20.5)
plt.ylabel('No. of Units',fontsize=20.5)
ax.tick_params(labelsize=15)
plt.title('Top 10 Vendors with Highest No. of Resources',fontsize=20.5)
for item in ax.get_xticklabels():
    item.set_rotation(90)
df4=pd.read_csv('../input/Teachers.csv')
df4.head(1)
df4['Teacher Prefix'].unique()
df4.dtypes
df4['Teacher First Project Posted Date']=pd.to_datetime(df4['Teacher First Project Posted Date'])
df4.head(1)
df4.isnull().sum()
df4['year_of_project']=df4['Teacher First Project Posted Date'].dt.year
df4['year_of_project'].unique()
cnt_srs = df4['year_of_project'].value_counts()

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color="blue",
        #colorscale = 'Blues',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Year-wise Projects Posted'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Proj")  
df5=pd.read_csv('../input/Projects.csv')
df5.head()
temp_series = df5['Project Resource Category'].value_counts()
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(
    title='Project Category Distribution',
    width=900,
    height=900,
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Projects")
cnt_srs = df5['Project Current Status'].value_counts()

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color=cnt_srs.values,
        colorscale = 'Picnic',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Funded & Non-Funded Projects'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="funding")
df5_fund_category=df5.groupby('Project Resource Category',as_index=False)['Project Cost'].agg('sum')
df5_fund_category.shape
fig, ax = plt.subplots()

fig.set_size_inches(11.7, 8.27)

sns.set_context("paper", font_scale=1.5)
f=sns.barplot(x=df5_fund_category["Project Resource Category"], y=df5_fund_category['Project Cost'], data=df5_fund_category)
f.set_xlabel("Category of Project",fontsize=15)
f.set_ylabel("Project Cost",fontsize=15)
f.set_title('Project Category-wise Cost')
for item in f.get_xticklabels():
    item.set_rotation(90)
df_funding=df5.ix[df5['Project Current Status']=="Fully Funded"]
df_funding.head()
cnt_srs = df_funding['Project Resource Category'].value_counts()

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color=cnt_srs.values,
        colorscale = 'Picnic',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Funded Project Resources Category-wise'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="funding_proj")
from PIL import Image
from wordcloud import WordCloud
import requests
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='black',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
).generate(str(data))

    fig = plt.figure(1, figsize=(15, 15))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()

show_wordcloud(df5['Project Title'])
df6=pd.read_csv('../input/Schools.csv')
df6.head()
df6.shape
temp_series = df6['School Metro Type'].value_counts()
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(
    title='School Metro Type Distribution',
    width=900,
    height=900,
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="School")
df6['School Name'].nunique()
df6=df6.sort_values('School Percentage Free Lunch',ascending=False)
cnt_srs = df6['School Percentage Free Lunch'].value_counts()
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=cnt_srs.values[::-1],
    orientation = 'h',
    marker=dict(
        reversescale = True,
        color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5,
                    ),
    ),
)

layout = dict(
    title='No. of schools giving a %  of Free Lunch',
     yaxis=dict(
        title='% of Free Lunch')
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Lunch")
