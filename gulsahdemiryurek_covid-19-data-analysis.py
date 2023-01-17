# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
d1=pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")

dconfirmed=pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

drecovered=pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")

ddeaths=pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")
d1.head()
dconfirmed.head()
dco=dconfirmed[dconfirmed["Country/Region"]=="Mainland China"]

dc=dco.drop(["Lat","Long","Country/Region"],axis=1)

dct=dc.T

dct.columns=dc["Province/State"]

dct.drop(dct.index[0],inplace=True)
import plotly.graph_objects as go

fig = go.Figure(data=go.Heatmap(

                   z=dct,

                   y=dct.index,

                   x=dct.columns,

                    colorscale="Blues",

                    hoverongaps=False))

fig.update_layout(

    title='Number of Confirmed Cases between 1/22/20 - 3/11/20 in China Regions')

fig.show()
dcr=drecovered[drecovered["Country/Region"]=="Mainland China"]

dcs=dcr.drop(["Lat","Long","Country/Region"],axis=1)

dcts=dcs.T

dcts.columns=dcs["Province/State"]

dcts.drop(dcts.index[0],inplace=True)
fig = go.Figure(data=go.Heatmap(

                   z=dcts,

                   y=dcts.index,

                   x=dcts.columns,

                    colorscale="Greens",

                    hoverongaps=False))

fig.update_layout(

    title='Number of Recovered Cases between 1/22/20 - 3/11/20 in China Regions')

fig.show()
dcd=ddeaths[ddeaths["Country/Region"]=="Mainland China"]

dcsd=dcd.drop(["Lat","Long","Country/Region"],axis=1)

dctsd=dcsd.T

dctsd.columns=dcsd["Province/State"]

dctsd.drop(dctsd.index[0],inplace=True)
fig = go.Figure(data=go.Heatmap(

                   z=dctsd,

                   y=dctsd.index,

                   x=dctsd.columns,

                    colorscale="Reds",

                    hoverongaps=False))

fig.update_layout(

    title='Number of Death Cases between 1/22/20 - 3/11/20 in China')

fig.show()
dx=d1.groupby(["Country/Region" ,"ObservationDate"]).sum()

dx.reset_index(inplace=True)

dchina=dx[dx["Country/Region"]=="Mainland China"]



fig = go.Figure()

fig.add_trace(go.Scatter(

    x=dchina.ObservationDate,

    y=dchina.Confirmed,

    name='Confirmed',

    mode='lines',

    marker_color='lightsalmon',

    marker=dict(size=10)

))

fig.add_trace(go.Scatter(

    x=dchina.ObservationDate,

    y=dchina.Recovered,

    name='Recovered',

    mode='lines',

    marker_color='yellowgreen',

    marker=dict(size=10)

))

fig.add_trace(go.Scatter(

    x=dchina.ObservationDate,

    y=dchina.Deaths,

    name='Deaths',

    mode='lines',

    marker_color='red',

    marker=dict(size=10)

    

))



fig.update_layout( xaxis_tickangle=-45)

fig.update_layout(

    title="Cumulative Number of Confirmed, Recovered and Death Cases in China ",

    xaxis_title="Observation Date",

    yaxis_title="Count",

    height=700,

    width=800,

    plot_bgcolor="white",

    paper_bgcolor='mistyrose'

)

fig.show()
da=dconfirmed[["Province/State","Lat","Long","Country/Region"]]

data=pd.merge(d1,da,on=["Province/State" ,"Country/Region"])

data.drop_duplicates(subset ="SNo", inplace = True) 

data["normConfirmed"]=  (data["Confirmed"] - np.min(data["Confirmed"]))/(np.max(data["Confirmed"])-np.min(data["Confirmed"]))
dskorea=dx[dx["Country/Region"]=="South Korea"]

dusa=dx[dx["Country/Region"]=="US"]

ditaly=dx[dx["Country/Region"]=="Italy"]

dfrance=dx[dx["Country/Region"]=="France"]

diran=dx[dx["Country/Region"]=="Iran"]

djapan=dx[dx["Country/Region"]=="Japan"]
from plotly.subplots import make_subplots

fig = make_subplots(rows=2, cols=3,subplot_titles=("South Korea","USA", "Italy","France","Iran","Japan"))



fig.add_trace(

    go.Scatter(x=dskorea.ObservationDate, y=dskorea.Confirmed,name="South Korea"),

    row=1, col=1

)



fig.add_trace(

    go.Scatter(x=dusa.ObservationDate, y=dusa.Confirmed,name="USA"),

    row=1, col=2

)

fig.add_trace(

    go.Scatter(x=ditaly.ObservationDate, y=ditaly.Confirmed,name="Italy"),

    row=1, col=3

)

fig.add_trace(

    go.Scatter(x=dfrance.ObservationDate, y=dfrance.Confirmed,name="France"),

    row=2, col=1

)

fig.add_trace(

    go.Scatter(x=diran.ObservationDate, y=diran.Confirmed,name="Iran"),

    row=2, col=2

)

fig.add_trace(

    go.Scatter(x=djapan.ObservationDate, y=djapan.Confirmed,name="Japan"),

    row=2, col=3

)



fig.update_layout(height=600, width=900, title_text="Cumulative Number of Confirmed Cases ",plot_bgcolor="white", paper_bgcolor='ivory')

fig.show()
fig = make_subplots(rows=2, cols=3,subplot_titles=("South Korea","USA", "Italy","France","Iran","Japan"))



fig.add_trace(

    go.Scatter(x=dskorea.ObservationDate, y=dskorea.Recovered,name="South Korea"),

    row=1, col=1

)



fig.add_trace(

    go.Scatter(x=dusa.ObservationDate, y=dusa.Recovered,name="USA"),

    row=1, col=2

)

fig.add_trace(

    go.Scatter(x=ditaly.ObservationDate, y=ditaly.Recovered,name="Italy"),

    row=1, col=3

)

fig.add_trace(

    go.Scatter(x=dfrance.ObservationDate, y=dfrance.Recovered,name="France"),

    row=2, col=1

)

fig.add_trace(

    go.Scatter(x=diran.ObservationDate, y=diran.Recovered,name="Iran"),

    row=2, col=2

)

fig.add_trace(

    go.Scatter(x=djapan.ObservationDate, y=djapan.Recovered,name="Japan"),

    row=2, col=3

)



fig.update_layout(height=600, width=900, title_text="Cumulative Number of Recovered Cases ",plot_bgcolor="white",paper_bgcolor='oldlace')

fig.show()
fig = make_subplots(rows=2, cols=3,subplot_titles=("South Korea","USA", "Italy","France","Iran","Japan"))



fig.add_trace(

    go.Scatter(x=dskorea.ObservationDate, y=dskorea.Deaths,name="South Korea"),

    row=1, col=1

)



fig.add_trace(

    go.Scatter(x=dusa.ObservationDate, y=dusa.Deaths,name="USA"),

    row=1, col=2

)

fig.add_trace(

    go.Scatter(x=ditaly.ObservationDate, y=ditaly.Deaths,name="Italy"),

    row=1, col=3

)

fig.add_trace(

    go.Scatter(x=dfrance.ObservationDate, y=dfrance.Deaths,name="France"),

    row=2, col=1

)

fig.add_trace(

    go.Scatter(x=diran.ObservationDate, y=diran.Deaths,name="Iran"),

    row=2, col=2

)

fig.add_trace(

    go.Scatter(x=djapan.ObservationDate, y=djapan.Deaths,name="Japan"),

    row=2, col=3

)



fig.update_layout(height=600, width=900, title_text="Cumulative Number of Deaths  ",plot_bgcolor="white",paper_bgcolor='honeydew')

fig.show()
import plotly.express as px



fig = px.density_mapbox(data, lat="Lat", lon="Long", hover_name="Province/State", hover_data=['normConfirmed',"Confirmed","Deaths","Recovered"],animation_frame="ObservationDate",

                        color_continuous_scale="Portland",radius=7,  zoom=0,height=700)

fig.update_layout(mapbox_style="carto-darkmatter", mapbox_center_lon=0)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})





fig.show()