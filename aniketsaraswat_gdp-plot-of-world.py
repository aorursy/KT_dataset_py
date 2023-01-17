import numpy as np
import pandas as pd
import chart_studio.plotly as py
import plotly.offline as po
import plotly.graph_objs as pg
%matplotlib inline
po.init_notebook_mode(connected=True)
gdp=pd.read_csv("../input/geoplotting/215.gdp.csv")
gdp.head()
data=dict(type='choropleth',
         locations=gdp['CODE'],
         z=gdp['GDP (BILLIONS)'],
         text=gdp['COUNTRY'],
         colorbar={'title':'Units'})
layout=dict(title="World GDP Plot",geo=dict(showframe=False,projection={'type':'stereographic'},coastlinecolor='blue',showlakes=True))
#type can be hammer,robinson,natural earth,mercator,stereographic
x=pg.Figure(data=[data],layout=layout)
po.iplot(x)
