import numpy as np 
import pandas as pd
import warnings
warnings.simplefilter("ignore")
import numpy as np
import pandas as pd
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
train = pd.read_csv('../input/city-of-seattle-bicycle-racks/city-of-seattle-bicycle-racks.csv', nrows = 30_000)
train.head()
print("Total Number of Bike Racks: {}".format(train.shape[0]))
# Adapted from https://www.kaggle.com/shaz13/simple-exploration-notebook-map-plots-v2/
data = [go.Scattermapbox(
            lat= train['LATITUDE'] ,
            lon= train['LONGITUDE'],
            mode='markers',
            text=train['UNITDESC'],
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
                                zoom=11,
                                center= dict(
                                         lat=47.6137,
                                         lon=-122.2772),
                               ),
                    width=1800,
                    height=1200, title = "Bike Racks in Seattle")
fig = dict(data=data, layout=layout)
iplot(fig)
# delete secret token
from IPython.display import clear_output; clear_output(wait=True) 
!rm -rf "../working/"