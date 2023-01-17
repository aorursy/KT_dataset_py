# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import geopandas

import pandas_profiling

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

'''Seaborn and Matplotlib Visualization'''

import matplotlib                  # 2D Plotting Library

import matplotlib.pyplot as plt

import seaborn as sns              # Python Data Visualization Library based on matplotlib

import geopandas as gpd            # Python Geospatial Data Library

plt.style.use('fivethirtyeight')

%matplotlib inline



'''Plotly Visualizations'''

import plotly as plotly                # Interactive Graphing Library for Python

import plotly.express as px

import plotly.graph_objects as go

from plotly.offline import init_notebook_mode, iplot, plot

init_notebook_mode(connected=True)



'''Spatial Visualizations'''

import folium

import folium.plugins



'''NLP - WordCloud'''

import wordcloud

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



'''Machine Learning'''

import sklearn

from sklearn import preprocessing

from sklearn import metrics

from sklearn.metrics import r2_score, mean_absolute_error

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression,LogisticRegression

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor 







# Any results you write to the current directory are saved as output.
import plotly.graph_objects as go

from plotly.offline import init_notebook_mode, iplot

import pandas as pd 



df = pd.read_csv("/kaggle/input/netflix-shows/netflix_titles.csv")

#by default head shows us top 5 records

df.head()
df.shape
df.info()
# Number of missing values in each column of training data

missing_val_count_by_column = (df.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
df["rating"].fillna("No rating", inplace = True) 
# Number of missing values in each column of training data

missing_val_count_by_column = (df.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
df.rename(columns={"type": "category"},inplace = True)
import pandas as pd



df = df[pd.notnull(df['director'])]
reduced_df = df.drop("director", axis=1)
sns.set(rc={'figure.figsize':(19.7,8.27)})



sns.heatmap(reduced_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.director)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200, background_color="coral").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.title('Most Popular Directors',fontsize = 30)

plt.axis("off")

plt.show()
import plotly.offline as pyo

import plotly.graph_objs as go

roomdf = reduced_df.groupby('release_year').size()/reduced_df['release_year'].count()*100

labels = roomdf.index

values = roomdf.values



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6)])

fig.show()
df_countries = pd.DataFrame(reduced_df.country.value_counts().reset_index().values, columns=["country", "count"])

df_countries.head()
iplot([go.Choropleth(

    locationmode='country names',

    locations=df_countries.country,

    z=df_countries["count"]

)])
#initializing empty list where we are going to put our name strings

_ratings_=[]

#getting name strings from the column and appending it to the list

for rating in reduced_df.rating:

    _ratings_.append(rating)

#setting a function that will split those name strings into separate words   

def split_rating(rating):

    spl=str(rating).split()

    return spl

#initializing empty list where we are going to have words counted

_rating_for_count_=[]

#getting name string from our list and using split function, later appending to list above

for x in _ratings_:

    for word in split_rating(x):

        word=word.lower()

        _rating_for_count_.append(word)

        

#we are going to use counter

from collections import Counter

#let's see top 5 used words by host to name their listing

_top_5_w=Counter(_rating_for_count_).most_common()

_top_5_w=_top_5_w[0:5]







#now let's put our findings in dataframe for further visualizations

sub_w=pd.DataFrame(_top_5_w)

sub_w.rename(columns={0:'Ratings', 1:'Count'}, inplace=True)
#we are going to use barplot for this visualization

# fig = px.bar(sub_w, x='Ratings', y='Count')

# fig.show()

fig = px.bar(sub_w, x="Ratings", y="Count", color='Ratings')

fig.show()
# x = reduced_df.duration.value_counts()

movie_df = reduced_df[reduced_df['category'] == 'Movie']

x = movie_df.rating.value_counts()

x.head()
TV_show_df = reduced_df[reduced_df['category'] == 'TV Show']

x1 = TV_show_df.rating.value_counts()

x1.head()
import plotly.graph_objects as go



fig = go.Figure()

fig.add_trace(go.Bar(

    y=['TV-MA','TV-14','TV-PG'],

    x=[1306, 1015,415],

    name='Movie',

    orientation='h',

    marker=dict(

        color='rgba(246, 78, 139, 0.6)',

        line=dict(color='rgba(246, 78, 139, 1.0)', width=3)

    )

))

fig.add_trace(go.Bar(

    y=['TV-MA','TV-14','TV-PG'],

    x=[46,46,20],

    name='TV Show',

    orientation='h',

    marker=dict(

        color='rgba(58, 71, 80, 0.6)',

        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)

    )

))



fig.update_layout(barmode='stack')

fig.show()
date = pd.DataFrame(reduced_df.date_added.value_counts().reset_index().values, columns=["Date", "Count"])

date.head()
import plotly.express as px



# df = px.data.gapminder().query("continent=='Oceania'")

fig = px.line(date, x="Date", y="Count",title = "Line graph showing amount of content added on Netflix date wise.")

fig.show()