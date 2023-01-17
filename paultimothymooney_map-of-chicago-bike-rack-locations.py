import numpy as np 
import pandas as pd
import warnings
warnings.simplefilter("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import plotly
import plotly.plotly as py
import plotly.offline as offline
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import Scatter, Figure, Layout
from scipy.special import boxcox
init_notebook_mode(connected=True)
import cufflinks as cf
# begin secret_token
from shutil import copyfile
copyfile(src = "../input/private-mapbox-access-token/private_mapbox_access_token.py", dst = "../working/private_mapbox_access_token.py")
from private_mapbox_access_token import *
private_mapbox_access_token = private_mapbox_access_token()
# end secret_token
cf.set_config_file(offline=True)
train = pd.read_csv('../input/chicago-bike-racks/bike-racks.csv')
#train.head()
print("Total Number of Bike Racks: {}".format(train.shape[0]))
target = train['Community Name']
data = [go.Histogram(x=target)]
layout = go.Layout(title = "Bike Racks per Neighborhood")
fig = go.Figure(data=data, layout=layout)
iplot(fig)
# adapted from https://www.kaggle.com/shaz13/simple-exploration-notebook-map-plots-v2/
data = [go.Scattermapbox(
            lat= train['Latitude'] ,
            lon= train['Longitude'],
            customdata = train['Address'],
            mode='markers',
            text=train['Address'],
            marker=dict(
                size= 4,
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
                                #style= "mapbox://styles/shaz13/cjiog1iqa1vkd2soeu5eocy4i"
                               ),
                    width=1800,
                    height=1200, title = "Bike Racks in Chicago")
fig = dict(data=data, layout=layout)
iplot(fig)
# delete secret token
from IPython.display import clear_output; clear_output(wait=True) # delete secret_token
!rm -rf "../working/"