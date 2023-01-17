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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import plotly as plotly

import seaborn as sns

from sklearn import preprocessing

import geopandas as gpd
from plotly import __version__

import plotly.offline as py 

from plotly.offline import init_notebook_mode, plot

init_notebook_mode(connected=True)

from plotly import tools

import plotly.graph_objs as go

import plotly.express as px

import folium

from folium.plugins import MarkerCluster

from folium import plugins

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
df_bnb = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

df_bnb.head()
print("The amount of listings of Airbnbs are: ", len(df_bnb))
df_bnb.dtypes
#Finding the number of null data points 

df_bnb.isnull().sum()
df_bnb.drop(['host_name','last_review'],axis = 1, inplace = True)

df_bnb.head()
df_bnb.fillna({'reviews_per_month': 0},inplace=True)

boroughs = df_bnb.neighbourhood_group.unique()

boroughs
hoods = df_bnb.neighbourhood.unique()

hoods