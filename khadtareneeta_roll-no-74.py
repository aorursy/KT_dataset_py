import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import seaborn as sns

import time

import warnings

warnings.filterwarnings('ignore')
global_temp_country = pd.read_csv('../input/GlobalLandTemperaturesByCountry.csv')
global_temp_country.describe()
global_temp_country['AverageTemperature'].hist(bins=10)
Mcity_Data = pd.read_csv('../input/GlobalLandTemperaturesByMajorCity.csv')

var=Mcity_Data['Latitude'].value_counts()
var
var.plot(kind='bar')