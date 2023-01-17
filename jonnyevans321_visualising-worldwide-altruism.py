#______________

# the packages

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode,iplot

init_notebook_mode(connected=True)

#___________________

# and the dataframe

dm = pd.read_csv("../input/CAF_donating_money.csv",encoding='latin-1')



#___________________________

# countries in the dataframe

dm.head()
data = dict(type='choropleth',

locations = dm['Country'],

locationmode = 'country names',

text = dm['Country'], colorbar = {'title':'Percentage'}, z=dm['Percentage'],

colorscale=[[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\

            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],    

reversescale = True)



layout = dict(title='Percentage of population that has donated money to charity',width=1000,margin=dict(

        l=0,

        r=0,

        b=0,

        t=50,

        pad=4

    ),geo = dict(showframe = False, projection={'type':'mercator'}))



choromap = go.Figure(data = [data], layout = layout)

iplot(choromap, validate=False)