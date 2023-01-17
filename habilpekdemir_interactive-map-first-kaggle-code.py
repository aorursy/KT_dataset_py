import numpy as np

import pandas as pd

import plotly.express as px

import plotly.graph_objs as go

import matplotlib.pyplot as plt

import seaborn as sns
from plotly.offline import download_plotlyjs , init_notebook_mode , plot , iplot
init_notebook_mode(connected=True)
%matplotlib inline
df = pd.read_csv('../input/pizza-restaurants-and-the-pizza-they-sell/8358_1.csv')
#df.info

#df.head(5)
top_pizza_type = pd.DataFrame(data=df['menus.name'].value_counts().head(10))
top_pizza_type['type'] = top_pizza_type.index
fig,ax = plt.subplots(figsize = (18,8))

sns.barplot(x='type' , y= 'menus.name' , data=top_pizza_type , ax=ax)
top_pizza_cities = df['city'].value_counts().head(15)

fig,ax = plt.subplots(figsize = (12,6))

top_pizza_cities.plot.bar()
fig = px.scatter_mapbox(df , lat='latitude' , lon='longitude',

      hover_name='name'  , hover_data=['menus.name' , 'menus.amountMax'],zoom = 3,height=300)
fig.update_layout(mapbox_style = "open-street-map")

fig.update_layout(margin = {'r':0 , 't':0 , 'l':0 , 'b':0})

fig.show()