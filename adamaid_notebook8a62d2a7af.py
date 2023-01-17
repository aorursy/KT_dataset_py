import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

from wordcloud import WordCloud, STOPWORDS

from scipy.misc import imread

import base64
# Load in the .csv files as three separate dataframes

Global = pd.read_csv('../input/global.csv') # Put to caps or else name clash

national = pd.read_csv('../input/national.csv')

regional = pd.read_csv('../input/regional.csv')
# Print the top 3 rows

regional.head(3)