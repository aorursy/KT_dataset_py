import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import pandas as pd

import numpy as np
from plotly.figure_factory import create_table

import plotly.express as px



gapminder = px.data.gapminder()



table = create_table(gapminder.head(10))

py.iplot(table)
type(gapminder)
data_canada = px.data.gapminder().query("country == 'Canada'")

fig = px.bar(data_canada, x='year', y='pop', height=400)

fig.show()
fig = px.bar(data_canada, x='year', y='pop',

             hover_data=['lifeExp', 'gdpPercap'], color='lifeExp',

             labels={'pop':'population of Canada'}, height=400)

fig.show()
gapminder2007 = gapminder.query("year == 2007")



px.scatter(gapminder2007, x="gdpPercap", y="lifeExp")
px.scatter(gapminder2007, x="gdpPercap", y="lifeExp", color="continent")
px.scatter(gapminder2007, x="gdpPercap", y="lifeExp", color="continent", size="pop", size_max=60)
px.scatter(gapminder2007, x="gdpPercap", y="lifeExp", color="continent", size="pop", size_max=60, hover_name="country")
px.scatter(gapminder2007, x="gdpPercap", y="lifeExp", color="continent", size="pop", size_max=60,

          hover_name="country", facet_col="continent", log_x=True)
fig = px.scatter(gapminder, x="gdpPercap", y="lifeExp", animation_frame="year", animation_group="country",

           size="pop", color="continent", hover_name="country", facet_col="continent",

           log_x=True, size_max=45, range_x=[100,100000], range_y=[25,90])

fig.show()
px.scatter(gapminder, x="gdpPercap", y="lifeExp",size="pop", size_max=60, color="continent", hover_name="country",

           animation_frame="year", animation_group="country", log_x=True, range_x=[100,100000], range_y=[25,90],

           labels=dict(pop="Population", gdpPercap="GDP per Capita", lifeExp="Life Expectancy"))
fig = px.line_geo(gapminder.query("year==2007"), locations="iso_alpha", color="continent", projection="orthographic")

fig.show()
px.choropleth(gapminder, locations="iso_alpha", color="lifeExp", hover_name="country", animation_frame="year",

              color_continuous_scale=px.colors.sequential.Plasma, projection="natural earth")
fig = px.choropleth(gapminder, locations="iso_alpha", color="lifeExp", hover_name="country", animation_frame="year", range_color=[20,80])

fig.show()
fig = px.line(gapminder, x="year", y="lifeExp", color="continent", line_group="country", hover_name="country",

        line_shape="spline", render_mode="svg")

fig.show()
fig = px.area(gapminder, x="year", y="pop", color="continent", line_group="country")

fig.show()