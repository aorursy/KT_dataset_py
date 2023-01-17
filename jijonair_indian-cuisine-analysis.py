#to run scenerio based on the below datasource
# 1 dishes by state wise
# 2 ingredients' wise dishes 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px
import itertools
import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
 #   for filename in filenames:
  #      print(os.path.join(dirname, filename))

data = pd.read_csv('/kaggle/input/indian-food-101/indian_food.csv')
read_data = pd.DataFrame(data)
#print(pd.DataFrame(read_data,columns=["name","ingredients"]))      
#ingredients = pd.DataFrame(read_data,columns=["ingredients"])
count_ingredients  = []
for x in data["ingredients"]:
   count_ingredients.append(len(x.split(',')))
read_data['ingredientscount'] = count_ingredients


fig = go.Figure()
sample = read_data.sort_values(by='name')
fig = go.Figure(data=go.Scatter(
    x = sample['name'],
    y = sample['ingredientscount'],
    mode='markers',
    marker=dict(
        size=10,
        color=sample['ingredientscount'], #set color equal to a variable
        colorscale='Plasma', # one of plotly colorscales
        showscale=True
    ),
    text= sample['name'],
))

fig.update_layout(title='Styled Scatter Plot (colored by Total Ingredients) - Dishes vs Total Ingredients',
                  xaxis_title='Name of Dishes',
                  yaxis_title='Total Number of Ingredients used',
                  paper_bgcolor='rgba(0,0,0,0)',
                  plot_bgcolor='rgba(0,0,0,0)',
                  font=dict(family='Cambria, monospace', size=12, color='#000000'))
fig.show()
distinct_ingredients  = []
flat_ingredients = []
for x in data["ingredients"]:
 distinct_ingredients.append(x.lower().strip().split(','))
flat_ingredients = list(itertools.chain(*distinct_ingredients))
#print(distinct_ingredients)

from collections import Counter
words_collection = Counter(flat_ingredients)
most_common = pd.DataFrame(words_collection.most_common(426))
most_common.columns = ['Most Used Ingredients','In Dishes']
pd.set_option("display.max_rows", None, "display.max_columns", None)

ingvsdish = go.Figure()
sample = most_common
ingvsdish = go.Figure(data=go.Scatter(
    x = most_common['Most Used Ingredients'],
    y = data['state'],
    mode='markers',
     marker=dict(
        size=10,
       color=most_common['In Dishes'], #set color equal to a variable
        colorscale='Plasma', # one of plotly colorscales
        showscale=True
    ),
    text=  most_common['Most Used Ingredients'],
))

ingvsdish.update_layout(title='Styled Scatter Plot (colored by Total Ingredients) - Dishes vs Total Ingredients',
                  xaxis_title='Name of Dishes',
                  yaxis_title='Total Number of Ingredients used',
                  paper_bgcolor='rgba(0,0,0,0)',
                  plot_bgcolor='rgba(0,0,0,0)',
                  font=dict(family='Cambria, monospace', size=12, color='#000000'))
ingvsdish.show()
