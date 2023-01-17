# Importing the libraries



import pandas as pd

import numpy as np

from datetime import datetime



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
# import ee; ee.Authenticate()
# !cat ~/.config/earthengine/credentials
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("API_KEY")
# Reading patient file



import pandas as pd

cor_patient = pd.read_csv("../input/coronavirusdataset/patient.csv")

cor_patient.head()
# Reading route file



cor_route = pd.read_csv("../input/coronavirusdataset/route.csv")

cor_route.head()
# Importing Bokeh libraries and functions



from bokeh.io import output_notebook, show, push_notebook

from bokeh.models import ColumnDataSource, GMapOptions

from bokeh.plotting import gmap

from bokeh.models.tools import HoverTool



output_notebook()
# Plotting all infection coordinates using route data via Bokeh library



map_options = GMapOptions(lat = 35.9078, lng = 127.7669, map_type = "roadmap", zoom = 7)

bokeh_plot = gmap(secret_value_0, map_options, title = "Infection Map")



source2 = ColumnDataSource(

    data = dict(lat = cor_route.latitude,

              lon = cor_route.longitude)

)



bokeh_plot.circle(x = "lon", y = "lat", size = 6,  color = "red", fill_alpha = 15, source = source2)



show(bokeh_plot)
# Reading time file



cor_time = pd.read_csv("../input/coronavirusdataset/time.csv")

cor_time.head()
# Plotting Tested vs Confirmed cases



plt.plot(cor_time['test'], cor_time['confirmed'])

plt.title('Tested vs Confirmed')

plt.xlabel('No. of Tested Patients')

plt.ylabel('Confirmed Results');

plt.show()



# Plotting frequency of testing



plt.plot(cor_time.index, cor_time['test'], 'r')

plt.title('Frequency of Testing')

plt.xlabel('Days')

plt.ylabel('No. of Tested Patients');

plt.show();