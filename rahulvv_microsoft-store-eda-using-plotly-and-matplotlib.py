# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('white')

import plotly.graph_objects as go

import plotly.express as px

import datetime

from plotly.offline import init_notebook_mode, iplot

from plotly.subplots import make_subplots

init_notebook_mode(connected=True)
data = pd.read_csv('/kaggle/input/windows-store/msft.csv', parse_dates=['Date'])

# Making 3 copies to use the original dataset on different occations

df = data.copy()

df1 = data.copy()

df_copy = data.copy()

data.head()
# Broadly classifying the ratings for convenience

mapping_dict = {

5.0:'Good',

4.5:'Good',    

4.0:'Good',    

3.5:'Average',     

3.0:'Average',     

2.5:'Average',     

2.0:'Poor',     

1.5:'Poor',      

1.0:'Poor',     

}



df.loc[:,'Rating'] = df.Rating.map(mapping_dict)

df.dropna(inplace=True)



# Chaning the price of paid apps to a single value-'Paid'.

for row in range(df.shape[0]):

    if df.loc[row, 'Price'] != 'Free':

        df.loc[row, 'Price'] = 'Paid'

df.tail()
plt.figure(figsize=(12,7))

g = sns.countplot(df.Rating, hue=df.Price, palette="cubehelix", order=['Good', 'Average', 'Poor'])

for p in g.patches:

    g.annotate(p.get_height(), (p.get_x() + p.get_width()/2, p.get_height()), ha='center', va='center', xytext=(0,10), textcoords = 'offset points')

g.set_ylabel('Total Ratings')

g.set_title('Frequency of Good, Average and Bad Ratings', fontsize=20)

plt.show()
fig = px.pie(df, names='Rating', color_discrete_sequence=px.colors.sequential.RdBu)

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.update_layout(title_text='Distribution of Ratings', font_size=15)

fig.show()
plt.figure(figsize=(12,7))

cat_df = df_copy.groupby(['Category'])['No of people Rated'].sum().sort_values(ascending=False).reset_index()

g = sns.barplot(x='Category', y='No of people Rated', data=cat_df, palette="cubehelix")

for p in g.patches:

  g.annotate(int(p.get_height()), (p.get_x() + p.get_width()/2, p.get_height()), ha='center', va='center', xytext=(0,10), textcoords = 'offset points')

g.set_xticklabels(g.get_xticklabels(), rotation=90)

g.set_title('Total number of people rated the Apps by Category', fontsize=20)

plt.show()
df_total_people = pd.DataFrame(df1.groupby(['Category'])['No of people Rated'].sum().reset_index())

df_avg_rating = pd.DataFrame(df1.groupby(['Category'])['Rating'].mean().reset_index())



df_final = df_total_people 

df_final['Average Rating'] = df_avg_rating['Rating']



fig = px.scatter(df_final, x='Average Rating', y='Category', size='No of people Rated', color='Category')

fig.update_layout(title="Average Rating - Category wise")

fig.show()
plt.figure(figsize=(12,7))

rate_df = df1.groupby(['Category'])['Rating'].mean().sort_values(ascending=False).reset_index()

g = sns.barplot(x='Category', y='Rating', data=rate_df, palette='GnBu_d')

for p in g.patches:

    g.annotate(format(p.get_height(), '0.2f'), (p.get_x() + p.get_width()/2, p.get_height()), ha='center', va='center', xytext=(0,10), textcoords = 'offset points')

g.set_xticklabels(g.get_xticklabels(), rotation=90)

g.set_title('Average Rating by Category', fontsize=20)

plt.show()
plt.figure(figsize=(12,7))

g = sns.boxplot(x='Category', y='Rating', data=data.dropna())

g.set_xticklabels(g.get_xticklabels(), rotation=90)

g.set_title('Interquartile distribution of Ratings by Category', fontsize=20)

plt.show()
percentage_free = format(df.Price.value_counts()['Free']/len(df)*100, '0.2f')

percentage_paid = format(df.Price.value_counts()['Paid']/len(df)*100, '0.2f')



colors = ['gold', 'mediumturquoise']



fig = go.Figure(data=[go.Pie(labels=['Free Apps','Paid Apps'],

                             values=[percentage_free, percentage_paid])])

fig.update_traces(hoverinfo='label+percent', textinfo='percent+label', textfont_size=20,

                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))

fig.update_layout(title_text='Distribution of Paid Apps & Free Apps', font_size=15)

fig.show()
df['Year'] = pd.DatetimeIndex(df['Date']).year

df_year = df.groupby(['Year', 'Category'])['No of people Rated'].sum()

df_year = pd.DataFrame(df_year.reset_index())

df_year
plt.figure(figsize=(16,9))

g = sns.barplot(x='Year', y='No of people Rated', data = df_year, palette='Set2', errwidth=1, capsize=0.5, dodge=True)

g.set_title('Total number of people rated the Apps by Year ', fontsize=20)

plt.show()
plt.figure(figsize=(20,9))

g = sns.barplot(x='Year', y='No of people Rated', data = df_year, hue='Category')

g.set_title('Total number of people rated the Apps by Year by Category ', fontsize=20)

g.legend(loc='upper left')

plt.show()
dayOfWeek={0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}

df['Weekday'] = pd.DatetimeIndex(df['Date']).weekday

df['Weekday'] = df['Weekday'].map(dayOfWeek)
table0 = pd.pivot_table(df, index='Weekday', values=['No of people Rated'], aggfunc={'No of people Rated':np.sum}).reset_index()

fig = px.bar(table0, x="Weekday", y="No of people Rated",

             color='Weekday',

             height=400)

fig.update_layout(title_text='No of people reviewed Apps by Weekday', font_size=15)

fig.show()
df_paid = df[df.Price=='Paid']

paid_perc_poor = format(df_paid.Rating.value_counts()['Poor']/df_paid.Rating.value_counts().sum()*100, '0.2f')

paid_perc_good = format(df_paid.Rating.value_counts()['Good']/df_paid.Rating.value_counts().sum()*100, '0.2f')

paid_perc_average = format(df_paid.Rating.value_counts()['Average']/df_paid.Rating.value_counts().sum()*100, '0.2f')
import plotly.graph_objects as go

colors = ['gold', 'mediumturquoise', 'lightgreen']

fig = go.Figure(data=[go.Pie(labels=['poor','good','average'],

                              values=[paid_perc_poor, paid_perc_good, paid_perc_average])])

fig.update_traces(hoverinfo='label+percent', textinfo='percent+label', textfont_size=20,

                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))

fig.update_layout(title_text='Distribution of Ratings of Paid Apps', font_size=15)

fig.show()
df_price = df1[df1.Price!='Free'].dropna()

df_price['price'] = df_price.Price.str.split(expand=True)[1]

df_price.drop('Price', axis=1, inplace=True)

df_price['price'] = df_price['price'].str.replace(',','')

df_price['price'] = df_price['price'].astype(float)

df_top_priced = df_price.sort_values(by='price', ascending=False).head(20)

df_top_priced = df_top_priced.reset_index(drop=True)

df_top_priced
fig = px.bar(df_top_priced, x='price', y='Name',

             hover_data=['Rating', 'No of people Rated'], color='Category',

             height=800, orientation='h', color_discrete_sequence=px.colors.qualitative.G10)

fig.update_layout(

    title="Most expensive Paid Apps - Category wise",

    xaxis_title="Pricing(₹)",

    yaxis_title=None

    )

fig.show()
table1 = pd.pivot_table(df_top_priced, index=['Category'], values=['No of people Rated'], aggfunc={'No of people Rated':np.sum})

table1 = table1.reset_index()

fig = px.bar(table1, x="Category", y="No of people Rated",

             color='Category',

             height=400)

fig.update_layout(title_text='No of people reviewed Paid Apps by Category', font_size=15)

fig.show()
table2 = pd.pivot_table(df_top_priced, index=['Category'], values=['price'], aggfunc={'price':np.mean})

table2 = table2.reset_index()

fig = px.bar(table2, x="Category", y="price",

             color='Category',

             height=400)

fig.update_layout(title_text='Average Price of Paid Apps by Category', font_size=15)

fig.show()

plt.figure(figsize=(12,7))

g = sns.countplot(df.Weekday, order = df.Weekday.value_counts().index, palette='GnBu_d')

g.set_xticklabels(g.get_xticklabels(), rotation=90)

g.set_title('Count of Apps published by Weekday', fontsize=20)

plt.show()