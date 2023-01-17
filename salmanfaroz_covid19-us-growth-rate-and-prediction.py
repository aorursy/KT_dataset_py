from pandas import read_csv, Grouper, DataFrame, concat

import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller

import statsmodels.tsa.api as smt

import numpy as np

from sklearn.metrics import mean_squared_error

import seaborn as sns

from datetime import datetime

from pandas_profiling import ProfileReport

from sklearn.preprocessing import MinMaxScaler

from keras.preprocessing import sequence 

from keras.models import Sequential 

from keras.layers import Dense, Embedding ,Dropout

from keras.layers import LSTM 

from plotly.offline import plot, iplot, init_notebook_mode

import plotly.graph_objs as go

import pandas as pd

import plotly.express as px

import plotly.figure_factory as ff

import datetime

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

from plotly.offline import init_notebook_mode, iplot

from urllib.request import urlopen

import json



init_notebook_mode(connected=True)  
ds=pd.read_csv("../input/us-counties-covid-19-dataset/us-counties.csv")

raw=ds.copy()


with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:

    counties = json.load(response)





fig = px.choropleth_mapbox(ds, geojson=counties, locations='fips', color='cases',

                           color_continuous_scale="Viridis",

                           range_color=(0, 20000),

                           mapbox_style="carto-positron",

                           hover_name ="county",

                           zoom=3, center = {"lat": 37.0902, "lon": -95.7129},

                           opacity=0.5

                          )



fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
ds['date']= pd.to_datetime(ds['date'])

ds.Timestamp=ds["date"]

ds.index = ds.Timestamp 

df = ds.resample('D').sum()
fig = go.Figure()



fig.add_trace(go.Scatter(

    x=df.index,

    y=df.cases,

    name="Cases in USA"     

))









fig.update_layout(

    font=dict(

        family="Courier New, monospace",

        size=18,

        color="RebeccaPurple"

    )

)



fig.show()
fig = go.Figure()



fig.add_trace(go.Scatter(

    x=df.index,

    y=df.deaths,

    name="Cases in USA"     

))









fig.update_layout(

    font=dict(

        family="Courier New, monospace",

        size=18,

        color="RebeccaPurple"

    )

)



fig.show()
case=df.cases

case=list(case)

population=case
growth_rate=[]



def growth():

    for pop in range(1, len(population)):

        gnumbers = (population[pop] / population[pop-1])/ population[pop-1]

        growth_rate.append(gnumbers)



growth()

def Average(growth_rate): 

    return sum(growth_rate) / len(growth_rate) 

  

average = Average(growth_rate) 



print("Now we have the Average growth factor of USA\n\n Growth factor value :",average)
val=df.iloc[-1]

val1=val["cases"]

ave='{0:.40f}'.format(average).rstrip("0") 

ave=float(ave)



result=val1*ave*60
print("Last day count is",val1,"\n\nGrowth factor is ",ave,"\n\n X is ",60,"\n")

print("After 60days case count wil be \n\n",result)

start = datetime.datetime.strptime("2020-08-15", "%Y-%m-%d")

end = datetime.datetime.strptime("2020-10-14", "%Y-%m-%d")

date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days)]



dt=[]



for date in date_generated:

    dt.append(date.strftime("%Y-%m-%d"))

dtd=pd.DataFrame()

dtd["Date"]=dt
fit1 = ExponentialSmoothing(np.asarray(df['cases']) ,seasonal_periods=7 ,trend='add', seasonal='add',).fit()

pred = fit1.forecast(60)

pred=pred.astype(int)

dtd["Holt"]=pred

dtd.Timestamp=dtd["Date"]

dtd.index = dtd.Timestamp 
fig = go.Figure()



fig.add_trace(go.Scatter(

    x=df.index,

    y=df.cases,

    name="Confirmed Cases"     

))



fig.add_trace(go.Scatter(

    x=dtd.index,

    y=dtd.Holt,

    name="Future Prediction"     

))







fig.update_layout(

    font=dict(

        family="Courier New, monospace",

        size=18,

        color="RebeccaPurple"

    )

)



fig.show()