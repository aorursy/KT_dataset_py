import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# visualiation tools
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
# import calmap
import folium
plt.style.use('ggplot')

# sci-kit learn tools
from scipy.stats import skew

# import random numbers
from random import random, randint   

# ignoreing sklearn warnings
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
# this will filter out a lot of future warnings from statsmodels
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# import library for Standardrize feature scales 
from sklearn.preprocessing import StandardScaler

sns.set_style('whitegrid')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
covid_19 = pd.read_csv("/kaggle/input/saudi-covid19-time-series-data/New_Saudi_Cities_COVID-19.csv")
covid_19.head()
covid_19.dtypes
covid_19['Date'] = pd.to_datetime(covid_19['Date'])
covid_19 = covid_19.sort_values(by="Date")
covid_19['Date'] = covid_19['Date'].dt.strftime('%m/%d/%Y')
covid_19.dtypes

import plotly.graph_objects as go

formated_gdf = covid_19.groupby(['Date', 'Country','City','Lat','Long'])['Cases', 'TotalDeaths'].max()
formated_gdf = formated_gdf.reset_index()
formated_gdf['size'] = formated_gdf['Cases'].pow(0.3)

fig = px.scatter_geo(formated_gdf, lat='Lat', lon='Long', locations = None, 
                     color="Cases", size='Cases', hover_name="City", 
                     range_color= [0, max(formated_gdf['Cases'])+2], animation_frame="Date", projection="natural earth",
                     color_discrete_sequence= px.colors.sequential.Plasma_r,
                     title='Spread over time')
# fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
# fig.update(layout_coloraxis_showscale=False)

# fig.write_("saudi_map.pdf")
fig.write_html("saudi_map.html")
# fig.write_image("saudi_map.png")