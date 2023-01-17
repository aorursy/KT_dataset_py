import pandas as pd
import numpy as np
import warnings
warnings.simplefilter("ignore")
# import plotly
import plotly.plotly as py
import plotly.graph_objs as go

# these two lines are what allow your code to show up in a notebook
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
rlcam_loc_data = pd.read_csv("../input/red-light-camera-locations.csv")
rlcam_loc_data.head()
rlcam_violations_data = pd.read_csv("../input/red-light-camera-violations.csv")
rlcam_violations_data.head()
print("Total Number of Red Light Camera Locations: {}" .format(rlcam_loc_data.shape[0]))
print("Total Number of Red Light Camera Violations: {}" .format(rlcam_violations_data.shape[0]))
null_columns=rlcam_loc_data.columns[rlcam_loc_data.isnull().any()]
rlcam_loc_data[null_columns].isnull().sum()
null_columns_violations=rlcam_violations_data.columns[rlcam_violations_data.isnull().any()]
rlcam_violations_data[null_columns_violations].isnull().sum()
print(rlcam_loc_data['FIRST APPROACH'].unique())
print(rlcam_loc_data['FIRST APPROACH'].value_counts())
data = [go.Bar(x= ['NB', 'SB', 'EB', 'WB', 'SEB'], y=rlcam_loc_data['FIRST APPROACH'].value_counts())]

iplot({'data': data,
       'layout': {
           'title': 'Value Counts',
           'xaxis': {
               'title': 'First Approach Value Counts'},
           'yaxis' : {
               'title': 'Counts'}
       }})
print(rlcam_loc_data['SECOND APPROACH'].unique())
print(rlcam_loc_data['SECOND APPROACH'].value_counts())
data = [go.Bar(x= ['WB', 'EB', 'SB', 'NB', 'SEB'], y=rlcam_loc_data['SECOND APPROACH'].value_counts())]

iplot({'data': data,
       'layout': {
           'title': 'Value Counts',
           'xaxis': {
               'title': 'SECOND Approach Value Counts'},
           'yaxis' : {
               'title': 'Counts'}
       }})
print(rlcam_loc_data['THIRD APPROACH'].unique())
print(rlcam_loc_data['THIRD APPROACH'].value_counts())
data = [go.Bar(x= ['NB', 'EB', 'SB'], y=rlcam_loc_data['THIRD APPROACH'].value_counts())]

iplot({'data': data,
       'layout': {
           'title': 'Value Counts',
           'xaxis': {
               'title': 'THIRD Approach Value Counts'},
           'yaxis' : {
               'title': 'Counts'}
       }})
data = [
    go.Scattermapbox(
        lat=rlcam_loc_data['LATITUDE'],
        lon=rlcam_loc_data['LONGITUDE'],
        mode='markers',
        marker=dict(
            size=11, 
            color = "rgb(180,31,33)"
        ),
        text=rlcam_loc_data['LOCATION'],
    )
]

layout = go.Layout(
    autosize=True,
    hovermode='closest',
    mapbox=dict(
        accesstoken="pk.eyJ1IjoiYW5pcnVkaDgzOCIsImEiOiJjanB3d2hmemwwZTB4NDNwOXlhendtYnBxIn0.zwms8s--1hpsc6KTyAtU6A",
        bearing=0,
        zoom = 10.3,
        center=dict(
            lat=41.895737080,
            lon=-87.6731264831),
    ),
        height = 1000,
        width = 1000,
        title = "Red Light Camera Locations"
)

fig = dict(data=data, layout=layout)
iplot(fig, filename='Red Light Cameras in Chicago')
