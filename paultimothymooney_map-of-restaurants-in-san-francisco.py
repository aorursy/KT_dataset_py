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
cf.set_config_file(offline=True)
# begin secret_token
from shutil import copyfile
copyfile(src = "../input/private-mapbox-access-token/private_mapbox_access_token.py", dst = "../working/private_mapbox_access_token.py")
from private_mapbox_access_token import *
private_mapbox_access_token = private_mapbox_access_token()
# end secret_token
train = pd.read_csv('../input/sf-restaurant-scores-lives-standard/restaurant-scores-lives-standard.csv')
train2 = train[train.risk_category == 'High Risk']
data = [go.Scattermapbox(
            lat= train['business_latitude'] ,
            lon= train['business_longitude'],
            mode='markers',
            text=train['business_name'],
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
                                zoom=12,
                                center= dict(
                                         lat=37.782852,
                                         lon=-122.388405),
                               ),
                    width=1800,
                    height=1200, title = "Restaurants in San Francisco")
fig = dict(data=data, layout=layout)
iplot(fig)
# Adapted from https://www.kaggle.com/shaz13/simple-exploration-notebook-map-plots-v2/
data = [go.Scattermapbox(
            lat= train['business_latitude'] ,
            lon= train['business_longitude'],
            mode='markers',
            text=train['violation_description'],
            marker=dict(
                size= 10,
                color = 'red',
                opacity = .8,
            ),
          )]
layout = go.Layout(autosize=False,
                   mapbox= dict(accesstoken=private_mapbox_access_token,
                                bearing=0,
                                pitch=0,
                                zoom=12,
                                center= dict(
                                         lat=37.782852,
                                         lon=-122.388405),
                               ),
                    width=1800,
                    height=1200, title = "Health Code Violations at Restaurants in San Francisco")
fig = dict(data=data, layout=layout)
iplot(fig)
# delete secret token
from IPython.display import clear_output; clear_output(wait=True)
!rm -rf "../working/"