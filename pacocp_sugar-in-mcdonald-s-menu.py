#all the imports I need

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import warnings

warnings.filterwarnings('ignore')
menu = pd.read_csv('../input/menu.csv')

menu.head(10)
df_sugars = pd.DataFrame(columns=('Item','Sugars'))

df_sugars['Item'] = menu['Item']

df_sugars['Sugars'] = menu['Sugars']

print("Let's sort them by the amount of sugar they have in a ascending order: ")

df_sugars = df_sugars.sort_values('Sugars', ascending=[True])

print(df_sugars.head(10))



print("Number of items in the menu: "+str(len(menu.index)))

print("Number of items without sugar in the menu: "+str(len(df_sugars.loc[df_sugars['Sugars'] == 0])))

print(df_sugars.loc[df_sugars['Sugars'] == 0])
print("Let's start with the bar chart")



data = [go.Bar(

             y = df_sugars['Sugars'].values,

            x = df_sugars['Item'].values,

    )]



py.iplot(data, filename='basic-bar')



# Now let's plot a scatter plot

# This plot is based on the one made by Anisotropic:

# https://www.kaggle.com/arthurtok/super-sized-we-mcdonald-s-nutritional-metrics



trace = go.Scatter(

    y = df_sugars['Sugars'].values,

    x = df_sugars['Item'].values,

    mode='markers',

    marker=dict(

        size= df_sugars['Sugars'].values,

        #color = np.random.randn(500), #set color equal to a variable

        color = df_sugars['Sugars'].values,

        colorscale='Portland',

        showscale=True

    ),

    text = menu['Item'].values

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'Scatter plot of Sugars per Item on the Menu',

    hovermode= 'closest',

    xaxis=dict(

        showgrid=False,

        zeroline=False,

        showline=False

    ),

    yaxis=dict(

        title= 'Sugars(g)',

        ticklen= 5,

        gridwidth= 2,

        showgrid=False,

        zeroline=False,

        showline=False

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatterChol')
# First let's add a new column to the dataframe, all equal to 50

df_sugars['Amount of Sugar recommended (g)'] = 50



# Let's plot them



trace1 = go.Bar(

    y = df_sugars['Sugars'].values,

    x = df_sugars['Item'].values,

    name='Sugars(g)'

)

trace2 = go.Bar(

    y = df_sugars['Amount of Sugar recommended (g)'].values,

    x = df_sugars['Item'].values,

    name='Recommended value of sugar OMS (g)'

)



data = [trace1, trace2]

layout = go.Layout(

    barmode='group'

)



layout= go.Layout(

    autosize= True,

    title= 'Relation between OMSs recommendation and  Sugars per Item on the Menu',

    hovermode= 'closest',

    xaxis=dict(

        showgrid=False,

        zeroline=False,

        showline=False

    ),

    yaxis=dict(

        title= 'Sugars(g)',

        ticklen= 5,

        gridwidth= 2,

        showgrid=False,

        zeroline=False,

        showline=False

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='grouped-bar')