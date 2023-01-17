import numpy as np 
import pandas as pd
import warnings
warnings.simplefilter("ignore")
from scipy.special import boxcox
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import plotly
import plotly.plotly as py
import plotly.offline as offline
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import cufflinks as cf
from plotly.graph_objs import Scatter, Figure, Layout
from IPython.core import display as ICD
cf.set_config_file(offline=True)
# begin secret_token
from shutil import copyfile
copyfile(src = "../input/private-mapbox-access-token/private_mapbox_access_token.py", dst = "../working/private_mapbox_access_token.py")
from private_mapbox_access_token import *
private_mapbox_access_token = private_mapbox_access_token()
# end secret_token
train = pd.read_csv('../input/chicago-beach-swim,-weather,-lab-data/beach-lab-data.csv', nrows = 30_000)
train2 = pd.read_csv('../input/chicago-beach-swim,-weather,-lab-data/beach-e.-coli-predictions.csv', nrows = 30_000)
train.head()
# Adapted from https://www.kaggle.com/shaz13/simple-exploration-notebook-map-plots-v2/
data = [go.Scattermapbox(
            lat= train['Latitude'] ,
            lon= train['Longitude'],
            mode='markers',
            text=train['Beach'],
            marker=dict(
                size= 10,
                color = 'black',
                opacity = .8,
            ),
          )]
layout = go.Layout(autosize=False,
                   mapbox= dict(accesstoken=private_mapbox_access_token,
                                bearing=0,
                                pitch=0,
                                zoom=10,
                                center= dict(
                                         lat=41.881900,
                                         lon=-87.325808),
                               ),
                    width=1800,
                    height=1200, title = "Map of Beaches in Chicago")
fig = dict(data=data, layout=layout)
iplot(fig)
target = train['Beach']
data = [go.Histogram(x=target)]
layout = go.Layout(title = "Number of E. coli Tests per Beach")
fig = go.Figure(data=data, layout=layout)
#iplot(fig)

eColi = train2.sort_values('Predicted Level',ascending=True)
eColi = eColi[['Beach Name','Predicted Level']]
inds = eColi.groupby(['Beach Name'])['Predicted Level'].transform(max) == eColi['Predicted Level']
eColi = eColi[inds]
eColi.reset_index(drop=True, inplace=True)
#print('Beaches in Chicago with the lowest predicted E. coli levels: ')
#ICD.display(eColi.head(10))

eColi2 = train2.sort_values('Predicted Level',ascending=False)
eColi2 = eColi2[['Beach Name','Predicted Level']]
inds = eColi2.groupby(['Beach Name'])['Predicted Level'].transform(max) == eColi2['Predicted Level']
eColi2 = eColi2[inds]
eColi.reset_index(drop=True, inplace=True)
#print('Beaches in Chicago with the highest predicted E. coli levels: ')
#ICD.display(eColi2.head(10))
trace1 = go.Bar(
                x = eColi['Beach Name'],
                y = eColi['Predicted Level'],
                name = "E. coli",
                marker = dict(color = 'rgba(0, 0, 255, 0.8)',
                             line=dict(color='rgb(128,0,0)',width=1.5)),
                text = eColi['Predicted Level'])
data = [trace1]
layout = go.Layout(barmode = "group",title='Maximum Predicted E. coli Level per Beach (ascending)', xaxis= dict(title= 'Beach Name',ticklen= 5,zeroline= False),yaxis= dict(title= 'Max Predict E. coli level (CFU/100ml)'))
fig = go.Figure(data = data, layout = layout)
iplot(fig)
# delete secret token
from IPython.display import clear_output; clear_output(wait=True) # delete secret_token
!rm -rf "../working/"
