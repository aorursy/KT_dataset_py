# import libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import ipywidgets

from ipywidgets import interact

from ipywidgets import interact_manual

import plotly.express as px

import plotly.graph_objects as go





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# upload dataset

data= pd.read_csv("/kaggle/input/movies-meta-data/movie_metadata.csv")



# dataset shape

print("shape of dataset:", data.shape)



# to see all the columns

pd.set_option("max_columns", 28)



data.head()
# dataset

reviews= data.groupby("director_name")["num_critic_for_reviews"].sum().sort_values(ascending=False).head(10)



# plot

plt.figure(figsize=(15,4))



plt.style.use('ggplot')

reviews.plot(kind="bar")

plt.title("Top 10 directors with maximum critic reviews")
df= data[["movie_title", "imdb_score"]][data["director_name"] == "Steven Spielberg"].sort_values(by= "imdb_score", ascending=False).head(10)

df



fig = px.bar(df, x='movie_title', y='imdb_score',

             color='imdb_score',

             

             text= "imdb_score",

             title= "Steven Spielberg top 10 movies",

             height=500)

fig.show()
fig = go.Figure(data=go.Scatter(x=data["num_voted_users"],

                                y=data["gross"],

                                mode='markers',

                                marker=dict(size=7, color=np.random.randn(5043),colorscale='Viridis',showscale=True)

                             )

               )

fig.update_layout(title_text='Number of voted users vs Gross', font_size=15,

                 xaxis = dict(

                               title_text = "Number of voted users",

                               title_font = {"size": 20},

                               title_standoff = 25),

                 yaxis = dict(

                              title_text = "Gross",

                              title_standoff = 25))

fig.show()
fig = go.Figure(data=go.Scatter(x=data["num_user_for_reviews"],

                                y=data["gross"],

                                mode='markers',

                                marker=dict(size=7, color=np.random.randn(5043),colorscale='Bluered_r',showscale=True)

                               )

               )

fig.update_layout(title_text='Number of user for reviews vs Gross', font_size=15,

                 xaxis = dict(

                               title_text = "Number of user for reviews",

                               title_font = {"size": 20},

                               title_standoff = 25),

                 yaxis = dict(

                              title_text = "Gross",

                              title_standoff = 25))



fig.show()
import plotly.graph_objects as go



fig = go.Figure(data=go.Scatter(x=data["imdb_score"],

                                y=data["gross"],

                                mode='markers',

                                marker=dict(size=7, color=np.random.randn(5043),colorscale='Viridis',showscale=True)

                               )

               )

fig.update_layout(title_text='Imdb Score vs Gross', font_size=15,

                 xaxis = dict(

                               title_text = "Imdb Score",

                               title_font = {"size": 20},

                               title_standoff = 25),

                 yaxis = dict(

                              title_text = "Gross",

                              title_standoff = 25))

fig.show()
import plotly.graph_objects as go



fig = go.Figure(data=go.Scatter(x=data["budget"],

                                y=data["gross"],

                                mode='markers',

                                marker=dict(size=7, color=np.random.randn(5043),colorscale='Inferno',showscale=True)

                               )

               )

fig.update_layout(title_text='Budget vs Gross', font_size=15,

                 xaxis = dict(

                               title_text = "Budget",

                               title_font = {"size": 20},

                               title_standoff = 25),

                 yaxis = dict(

                              title_text = "Gross",

                              title_standoff = 25))

fig.show()
data[(data["budget"]> 200000000) & (data["gross"]< 200000000)][[ "movie_title", "budget","gross", "country"]]
# prepare dataset

data2= data.groupby("country")[["budget"]].sum().sort_values(by= "budget", ascending=False).head(10)



# plot

fig=go.Figure(data= [go.Pie(labels = data2.index,

                            values = data2["budget"],

                            hole=0.4, pull= [0.05,0], opacity=0.9,

                            texttemplate = "%{label}: %{percent}",

                            textposition = "inside")],

                           

                                                        

              layout = go.Layout(title= "Top 10 countries with Total Movie Budget",

                                  font_size=12

                                 ))

fig.show()
@interact



def plot(x= list(data.select_dtypes(include="object").columns),

         y= list(data.select_dtypes(include="number").columns)

        ):

        

         return sns.barplot(data[x], data[y]) 

         plt.xticks(rotation=90)
# function to recommend movie based on language

def recommend_lang(x):

    y= data[["language", "movie_title", "imdb_score"]][data["language"] == x]

    y= y.sort_values(by= "imdb_score", ascending=False)

    return y
# Hindi recommended movis with highest imdb score

recommend_lang("Hindi")
# function to recommend movie based on language

def recommend_actor(x):

    a= data[["movie_title", "imdb_score"]][data["actor_1_name"] == x]

    b= data[["movie_title", "imdb_score"]][data["actor_2_name"] == x]

    c= data[["movie_title", "imdb_score"]][data["actor_3_name"] == x]

    

    x= a.append(b)

    y= x.append(c)

    y= y.sort_values(by= "imdb_score", ascending=False)

    return y
recommend_actor("Katrina Kaif")