import numpy as np 

import pandas as pd

from sklearn.preprocessing import LabelEncoder





#graph

import plotly.graph_objs as go

import plotly.express as px



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

df = pd.read_csv('../input/mushroom-classification/mushrooms.csv')
df.isnull().sum().sort_values(ascending=False)
df.head(8)
df.describe()
df_encoded = df.copy()

le = LabelEncoder()

for col in df_encoded.columns:

    df_encoded[col]=le.fit_transform(df_encoded[col])

    

df_encoded.head()
labels = ['Edible', 'Poison']

values = df['class'].value_counts()



fig=go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,

                  marker=dict(colors=['#87CEFA', '#7FFF00'],

                              line=dict(color='#FFFFFF',width=3)))

fig.show()
labels = ['Woods', 'Grasses', 'Paths', 'Leaves', 'Urban', 'Meadows', 'Waste']

values = df['habitat'].value_counts()

colors = ['#DEB887','#778899', '#B22222', '#FFFF00', 

          '#F8F8FF','#FFE4C4','#FF69B4']



fig=go.Figure(data=[go.Pie(labels=labels,

                           values=values,

                           #marker_colors=labels,

                           pull=[0.1, 0, 0, 0, 0.2, 0, 0])])

fig.update_traces(title='Mushrooms Habitat Percentage',

                  hoverinfo='label+value', 

                  textinfo='percent', 

                  opacity=0.9,

                  textfont_size=20,

                  marker=dict(colors=colors,

                             line=dict(color='#000000', width=0.1)),

                 )

fig.show()
labels = ['Brown', 'Gray', 'Red', 'Yellow', 'White', 'Buff', 'Pink', 

          'Cinnamon', 'Purple', 'Green']

values = df['cap-color'].value_counts()

colors = ['#DEB887','#778899', '#B22222', '#FFFF00', 

          '#F8F8FF','#FFE4C4','#FF69B4','#F0DC82','#C000C5', '#228B22']



fig=go.Figure(data=[go.Pie(labels=labels,

                           values=values,

                           #marker_colors=labels,

                           pull=[0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0])])

fig.update_traces(title='Mushrooms Color Quantity',

                  hoverinfo='label+percent', 

                  textinfo='value',

                  opacity=0.9,

                  textfont_size=20,

                  marker=dict(colors=colors,

                             line=dict(color='#000000', width=0.1)),

                 )

fig.show()
colors = ['#DEB887','#f8f8ff','#778899', '#FF69B4','#FFFF00','#B22222','#FFE4C4','#F0DC82','#C000C5', '#228B22']

fig = px.histogram(df, x='cap-color',

                   color_discrete_map={'p':'#7FFF00'},

                   #opacity=0.8,

                   color_discrete_sequence=[colors],

                   #barmode='relative',

                   barnorm='percent',

                   color='class'

                  )

fig.update_layout(title='Edible or Poisonous Percent Based on Cap Color',

                  xaxis_title='Cap Color',

                  yaxis_title='Quantity',

                 )



fig.show()
fig = px.histogram(df, x='odor',

                   color_discrete_map={'p':'#7FFF00', 'e':'#87CEFA'},

                   #opacity=0.8,

#                    color_discrete_sequence='Green',

                   barmode='group',

#                    barnorm='percent',

                   color='class'

                  )

fig.update_layout(title='Edible or Poisonous vs Odor',

                  xaxis_title='Cap Color',

                  yaxis_title='Quantity',

                  #title_x=0.5,

                 )



fig.show()
fig = px.histogram(df, x='cap-shape',

                   color_discrete_map={'p':'#7FFF00', 'e':'#87CEFA'},

                   #opacity=0.8,

                   color_discrete_sequence=[colors],

                   barmode='group',

                   #barnorm='percent',

                   color='class'

                  )

fig.update_layout(

                  xaxis_title='Cap Shape',

                  yaxis_title='Quantity',

                  #title_x=0.5,

                 )



fig.show()
fig = px.sunburst(df, path=['class','bruises', 'population'],

                 color='class',

                 color_discrete_map={'e':'#87CEFA', 'p':'#7FFF00', 't':'red'},

                 )

fig.update_layout(title="Class vs Bruises vs Population",

                  title_x=0.5,

                  )



fig.update_traces(marker=dict(line=dict(width=3)))

fig.show()
gill_cats = ['class', 'gill-size', 'gill-attachment', 'gill-spacing', 'gill-color']

gill_data = df_encoded[gill_cats]

gill_corr = gill_data.corr() 

# gill_corr['class'].sort_values(ascending=False)



fig = px.imshow(gill_corr,

                color_continuous_scale = 'Greens',

                color_continuous_midpoint=0,

               )

fig.update_layout(title="Gill Categories Correlation Matrix")

fig.show()
stalk_cats = ['class', 'stalk-shape', 'stalk-color-below-ring', 'stalk-color-above-ring', 

              'stalk-surface-below-ring', 'stalk-surface-above-ring', 'stalk-root']

stalk_data = df_encoded[stalk_cats]

stalk_corr = stalk_data.corr() 

# stalk_corr['class'].sort_values(ascending=False)



fig = px.imshow(stalk_corr,

                color_continuous_scale = 'purples',

                color_continuous_midpoint=0,

               )

fig.update_layout(title="Stalk Categories Correlation Matrix")

fig.show()
corr_matrix = df_encoded.corr()

corr_matrix['class'].sort_values(ascending=False)