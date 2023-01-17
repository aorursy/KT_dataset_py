import plotly as py

import plotly.graph_objs as go

import pandas as pd

import numpy as np
import plotly.express as px

from plotly.figure_factory import create_table
gapminder = px.data.gapminder()



table = create_table(gapminder.head(10))

table
data_can = px.data.gapminder().query("country == 'Canada'")

fig = px.bar(data_can, x = 'year', y = 'pop', height = 500 )

fig.show()
fig = px.bar(data_can, x='year', y='pop', 

            hover_data=['lifeExp','gdpPercap'],color='lifeExp',

            labels={'pop':'population of canada'},height = 500)

fig.show()
data_GDP2007 = gapminder.query("year == 2007" )



px.scatter(data_GDP2007,x="gdpPercap",y="lifeExp")
px.scatter(data_GDP2007,x="gdpPercap",y="lifeExp",color='continent')
px.scatter(data_GDP2007,x="gdpPercap",y="lifeExp",color='continent',

          size='pop',size_max=60)
px.scatter(data_GDP2007,x="gdpPercap",y="lifeExp",color='continent',

          size='pop',size_max=60,hover_name="country")
px.scatter(data_GDP2007,x="gdpPercap",y="lifeExp",color='continent',

          size='pop',size_max=60,hover_name="country", facet_col = "continent",log_x=True)
px.scatter(gapminder,x="gdpPercap",y="lifeExp",color='continent',

          size='pop',size_max=60,hover_name="country", animation_frame="year",

          animation_group="country", log_x=True, range_x=[100,100000],range_y=[25,90],

          labels={'pop':"Population", 'gdpPercap':"GDP per Capita", 'lifeExp':"Life Expecentcy"})
px.choropleth(gapminder, locations="iso_alpha",color="lifeExp",hover_name="country",

             animation_frame="year",color_continuous_scale=px.colors.sequential.Plasma,

             projection="natural earth")
px.choropleth(gapminder, locations="iso_alpha",color="lifeExp",hover_name="country",

             animation_frame="year",color_continuous_scale=px.colors.sequential.Plasma,

             projection="orthographic")