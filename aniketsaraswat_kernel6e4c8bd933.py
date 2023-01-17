import plotly
import pandas as pd
import chart_studio.plotly as py
import numpy as np
import plotly.offline as po
import plotly.graph_objs as pg
%matplotlib inline
po.init_notebook_mode(connected=True)
dat=pd.read_csv("../input/geoplotting/214.agri.csv")
dat.head()
data=dict(type='choropleth',
          locations=dat['code'],locationmode='USA-states',z=dat['total exports'],text=dat['text'],
         colorscale='Portland',
         colorbar={'title':'Color scale'})
layout=dict(title="AGRI PLOT FOR STATES OF USA",geo=dict(scope='usa',showlakes=True))
x=pg.Figure(data=[data],layout=layout)
po.iplot(x)
