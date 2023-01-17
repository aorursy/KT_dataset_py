import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use('ggplot')

from wordcloud import WordCloud

#Imports required for Plotly

import plotly.graph_objs as go

import plotly.offline as py

import plotly.figure_factory as ff

from plotly import tools

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True) #This is important 

import plotly_express as px

#Plotly Imports Ends

import warnings

warnings.filterwarnings('ignore')

import os

print(os.listdir("../input"))

rest_data=pd.read_csv('../input/zomato.csv')

print('We have a total of {0} restaurants in the data set'.format(rest_data.shape[0]))
#vizualizing the first 5 observations

rest_data.head()
del rest_data['url']

del rest_data['address']

del rest_data['phone']

del rest_data['location']
rest_data.rename(columns={'listed_in(city)': 'Suburb','listed_in(type)': 'restaurant_type'}, inplace=True)
rest_data.info()
rest_data.rate = rest_data.rate.replace(np.nan, 'Newly Opened')
trace1 = go.Bar(

                x = rest_data.Suburb.value_counts().keys(),

                y = rest_data.Suburb.value_counts(),

                name = "Suburb",

#                 marker = dict(

#                          colorscale='Jet',

#                          showscale=True),

                text = rest_data.Suburb)

data1 = [trace1]

layout = go.Layout(title = 'Restaurant Distribution by Suburb', 

                   barmode = "group", 

                   yaxis=dict(title= 'Number of Restaurants'))

fig = go.Figure(data = data1, layout = layout)

py.offline.iplot(fig, filename = 'basic-line')
trace1 = go.Bar(

                x = rest_data['restaurant_type'].value_counts().keys(),

                y = rest_data['restaurant_type'].value_counts(),

                name = "restaurant_type",

#                 marker = dict(

#                          colorscale='Jet',

#                          showscale=True),

                text = rest_data['restaurant_type'])

data1 = [trace1]

layout = go.Layout(title = 'Restaurant Distribution by Type', 

                   barmode = "group", 

                   yaxis=dict(title= 'Number of Restaurants'))

fig = go.Figure(data = data1, layout = layout)

py.offline.iplot(fig, filename = 'basic-line')
trace1 = go.Bar(

                x = rest_data['rest_type'].value_counts().head(15).keys(),

                y = rest_data['rest_type'].value_counts().head(15),

                name = "rest_type",

#                 marker = dict(

#                          colorscale='Jet',

#                          showscale=True),

                text = rest_data['rest_type'])

data1 = [trace1]

layout = go.Layout(title = 'Restaurant Distribution by Sub-Categories', 

                   barmode = "group", 

                   yaxis=dict(title= 'Number of Restaurants'))

fig = go.Figure(data = data1, layout = layout)

py.offline.iplot(fig, filename = 'basic-line')
x = rest_data['online_order'].value_counts()

trace = go.Pie(labels = x.index, values = x)

layout = go.Layout(title = "Online Order")

fig = go.Figure(data=[trace], layout = layout)

py.iplot(fig, filename='pie_OnlineOrder')
x = rest_data['book_table'].value_counts()

trace = go.Pie(labels = x.index, values = x)

layout = go.Layout(title = "Book Table")

fig = go.Figure(data=[trace], layout = layout)

py.iplot(fig, filename='pie_bookTable')
trace = go.Scatter(y = rest_data['approx_cost(for two people)'], text = rest_data['name'], mode = 'markers', x = rest_data['rate'].apply(

                    lambda x: x.split('/')[0])

                   )

data1 = [trace]

layout = go.Layout(title='Cost v/s Ratings', xaxis = dict(title='Restaurant Rating'), yaxis = dict(title='Cost for two'))

fig = go.Figure(data = data1 , layout = layout)

py.iplot(fig, filename='pie_bookTable')
#Removing the '/5' suffix from restaurant ratings.

#Also removing comma from "approx cost for two people" which is amount in Indian Rupees.

rest_data['rate'] = rest_data['rate'].apply(lambda x: (x.split('/')[0]))

rest_data['approx_cost(for two people)'] = rest_data['approx_cost(for two people)'].str.replace(',','').astype(float)
px.scatter(rest_data, x="cuisines", y="approx_cost(for two people)")
rest_data['approx_cost(for two people)'].dropna(inplace=True)
#Commented this since kaggle was behaving weirdly at the time of committing the kernel with this cell

#px.scatter(rest_data, x="Suburb", y="rate", hover_name='name')

# trace = go.Scatter(y = rest_data['rate'], text = rest_data['name'], mode = 'markers', x = rest_data['Suburb'])

                   

# data1 = [trace]

# layout = go.Layout(title='Rating Distribution for each Suburban area', xaxis = dict(title='Suburb'), yaxis = dict(title='Rating'))

# fig = go.Figure(data = data1 , layout = layout)

# py.iplot(fig, filename='pie_bookTable')
c1 = ''.join(str(rest_data['dish_liked'].values))

from wordcloud import WordCloud

plt.figure(figsize=(10,10))

wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,

                      width=1500, height=1500).generate(c1)

plt.imshow(wordcloud)

plt.axis("off")

c2 = ''.join(str(rest_data['cuisines'].values))

from wordcloud import WordCloud

plt.figure(figsize=(10,7))

wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,

                      width=1000, height=1000).generate(c2)

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")
chain = rest_data['name'].value_counts().head(25).to_frame()

chain['Restaurant Names'] = chain.index

chain.rename(columns={'name':'Count'}, inplace=True)
px.bar(chain, x = 'Restaurant Names', y = 'Count')
df_1=rest_data.groupby(['Suburb','rest_type']).agg('count')

data=df_1.sort_values(['name'],ascending=False).groupby(['Suburb'],

                as_index=False).apply(lambda x : x.sort_values(by="name",ascending=False).head(5))['name'].reset_index().rename(columns={'name':'count'})
rest_data.groupby(['Suburb','rest_type']).agg('count').sort_values(['name'],ascending=False).groupby('Suburb').head(1).name
quick_bite = rest_data[rest_data['rest_type'] == 'Quick Bites'].name.value_counts().head(20).to_frame()

quick_bite['Restaurant Name'] = quick_bite.index

quick_bite.rename(columns={'name':'Count'}, inplace=True)

px.bar(quick_bite, x='Restaurant Name', y='Count')
cuisines = rest_data['cuisines'].value_counts().head(10).to_frame()

cuisines['Cuisine names'] = cuisines.index

px.bar(cuisines, x="Cuisine names", y="cuisines")