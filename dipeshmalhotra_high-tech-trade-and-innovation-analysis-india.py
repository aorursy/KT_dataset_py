import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

import plotly.io as pio

from plotly.offline import init_notebook_mode

pio.templates.default = "plotly_white"

# Doing this to make sure the graphs are visible in the kaggle kernels and not just a blank white screen

init_notebook_mode()



%matplotlib inline

plt.rcParams.update({'font.size':22})
te=pd.read_csv("../input/science-and-technology-in-india/science-and-technology-indicators-for-india-1.csv")
te[:5]
te.info()
te.drop([0],inplace=True)

te['Year']=te['Year'].astype(int)

te['Value']=te['Value'].astype('float32')
te['Indicator Name'].value_counts()
te['Indicator Code'].value_counts()
fig = px.scatter(te[te['Indicator Name']=="High-technology exports (% of manufactured exports)"], x="Year", y="Value",size='Value',log_x=True, size_max=60)

fig.show()
te.loc[te['Indicator Name']=='High-technology exports (% of manufactured exports)','Value']=te.loc[te['Indicator Name']=='High-technology exports (% of manufactured exports)','Value']**10
fig=px.scatter(te[te['Indicator Name'].isin(["High-technology exports (current US$)","High-technology exports (% of manufactured exports)"])],x="Year", y="Value",color='Indicator Name',size='Value',log_x=True, size_max=60,width=800,height=600)

fig.show()
temp=te[te['Indicator Code'].isin(('IP.PAT.NRES','IP.PAT.RESD'))]

fig=px.line(temp,x='Year',y='Value',color='Indicator Name')

fig.show()
temp2=te[te['Indicator Code'].isin(('IP.TMK.TOTL','IP.TMK.RESD','IP.TMK.NRES'))]

fig=px.line(temp2,x='Year',y='Value',color='Indicator Name')

fig.show()
t3=te[te['Indicator Code'].isin(('SP.POP.SCIE.RD.P6','SP.POP.TECH.RD.P6'))]  

fig=px.line(t3,x='Year',y='Value',color='Indicator Name')

fig.show()
fig=px.line(te[te['Indicator Code']=='GB.XPD.RSDV.GD.ZS'],x='Year',y='Value',color='Indicator Name')

fig.show()
fig=px.line(te[te['Indicator Code']=='IP.JRN.ARTC.SC'],x='Year',y='Value',color='Indicator Name')

fig.show()