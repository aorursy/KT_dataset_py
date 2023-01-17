import matplotlib.pyplot as plt
import matplotlib
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
import seaborn as sns 
import numpy as np
import pandas as pd
import numpy as np
import random as rnd
import re
import io
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from numpy import genfromtxt
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score , average_precision_score
from sklearn.metrics import precision_score, precision_recall_curve
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS
plt.style.use('fivethirtyeight')
%matplotlib inline

from mpl_toolkits.basemap import Basemap
from matplotlib import animation, rc
from IPython.display import HTML

import warnings
warnings.filterwarnings('ignore')

import base64
from IPython.display import HTML, display
import warnings
warnings.filterwarnings('ignore')
from scipy.misc import imread
import codecs
from subprocess import check_output
import folium 
from folium import plugins
from folium.plugins import HeatMap
import os

loan_data = pd.read_csv('../input/kiva_loans.csv', parse_dates=['date'])
theme_ids = pd.read_csv('../input/loan_theme_ids.csv')
theme_regions = pd.read_csv('../input/loan_themes_by_region.csv')
mpi_region = pd.read_csv('../input/kiva_mpi_region_locations.csv')
# Any results you write to the current directory are saved as output.
loan_data.head(5)
f, ax = plt.subplots(figsize=(15, 20)) 
sns.barplot( y = loan_data['country'].value_counts().index,
            x = loan_data['country'].value_counts().values,
                palette="GnBu_d")
ax.set_ylabel('')
ax.set_title( 'Countries availing loan' );
f, ax = plt.subplots(figsize=(15, 15)) 
sns.barplot( y = loan_data['sector'].value_counts().index,
            x = loan_data['sector'].value_counts().values,
                palette="GnBu_d")
ax.set_ylabel('')
ax.set_title( 'Loan usage' );
loan_data['Century'] = loan_data.date.dt.year
loan_data_raw = loan_data.groupby(['country', 'Century'])['loan_amount'].mean().unstack()
loan_data_raw = loan_data_raw.sort_values([2017], ascending=False)
f, ax = plt.subplots(figsize=(15, 20)) 
loan_data_raw = loan_data_raw.fillna(0)
g = sns.heatmap(loan_data_raw,cmap='YlGnBu',linewidths=.5,vmin=0.01)
plt.show()
loan_data['Century'] = loan_data.date.dt.year
loan_data_raw = loan_data.groupby(['country', 'Century'])['loan_amount'].sum().unstack()
loan_data_raw.columns.name = None      
loan_data_raw = loan_data_raw.reset_index()  

year = [2014,2015,2016,2017]

for k in year :
    data =  dict(
            type = 'choropleth',
            locations = loan_data_raw['country'],
            locationmode = 'country names',
            z = loan_data_raw[k],
            text = loan_data_raw['country'],
            colorbar = {'title': 'Loan Amount,$'})

    layout = dict( title = 'Loan amount availed by countries '+ str(k)+ 'th century',
             geo = dict(showframe = False,
             projection = {'type' : 'Mercator'}))

    choromap3 = go.Figure(data = [data],layout=layout)
    iplot(choromap3)
loan_data_raw = loan_data.groupby(['sector', 'Century'])['loan_amount'].count().unstack()
loan_data_raw = loan_data_raw.sort_values([2017], ascending=False)
f, ax = plt.subplots(figsize=(15, 20)) 
loan_data_raw = loan_data_raw.fillna(0)
g = sns.heatmap(loan_data_raw,annot=True,fmt="2.0f",cmap='YlGnBu',linewidths=.5,vmin=0.01)
plt.show()
loan_data_raw = loan_data.groupby(['sector', 'Century'])['loan_amount'].sum().unstack()
loan_data_raw = loan_data_raw.sort_values([2017], ascending=False)
f, ax = plt.subplots(figsize=(15, 20)) 
loan_data_raw = loan_data_raw.fillna(0)
g = sns.heatmap(loan_data_raw,annot=True,fmt="2.0f",cmap='YlGnBu',linewidths=.5,vmin=0.01)
plt.show()
df = pd.DataFrame(loan_data['repayment_interval'].value_counts().values,
                  index=loan_data['repayment_interval'].value_counts().index, 
                  columns=[' '])

df.plot(kind='pie', subplots=True, autopct='%1.0f%%', figsize=(8, 8))
#plt.subplots_adjust(wspace=0.5)
plt.show()
loan_data_filter = loan_data[loan_data['repayment_interval'] == "irregular"]
loan_data_raw = loan_data_filter.groupby(['sector', 'Century'])['loan_amount'].count().unstack()
loan_data_raw = loan_data_raw.sort_values([2017], ascending=False)
f, ax = plt.subplots(figsize=(15, 20)) 
loan_data_raw = loan_data_raw.fillna(0)
g = sns.heatmap(loan_data_raw,annot=True,fmt="2.0f",cmap='YlGnBu',linewidths=.5,vmin=0.01)
plt.show()
loan_data_filter = loan_data[loan_data['repayment_interval'] == "monthly"]
loan_data_raw = loan_data_filter.groupby(['sector', 'Century'])['loan_amount'].count().unstack()
loan_data_raw = loan_data_raw.sort_values([2017], ascending=False)
f, ax = plt.subplots(figsize=(15, 20)) 
loan_data_raw = loan_data_raw.fillna(0)
g = sns.heatmap(loan_data_raw,annot=True,fmt="2.0f",cmap='YlGnBu',linewidths=.5,vmin=0.01)
plt.show()