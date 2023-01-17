#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTFB5FUzbkBrIpkyouf_8hnjGdbSGgON-zsKYN4Vpei-0kaY1WM&usqp=CAU',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
nRowsRead = 1000 # specify 'None' if want to read whole file

df = pd.read_csv('../input/cusersmarildownloadsdepressioncsv/depression.csv', delimiter=';', encoding = "ISO-8859-1", nrows = nRowsRead)

df.dataframeName = 'depression.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')
df.head()
fig = px.treemap(df, path=['emad'], values='epad',

                  color='epad', hover_data=['emad'],

                  color_continuous_scale='Rainbow')

fig.show()
ax = df.plot(figsize=(15,8), title='Depression Medication')

ax.set_xlabel('epad,epan,ewad,ewan,emad,eman')

ax.set_ylabel('Count')
df.iloc[0]
df.plot.hist()
df.plot(kind = 'hist', stacked = False, bins = 100)
from pandas.plotting import scatter_matrix

scatter_matrix(df, figsize= (8,8), diagonal='kde', color = 'b')

plt.show()
import seaborn as sns

def plot_correlation_map( df ):

    corr = df.corr()

    _ , ax = plt.subplots( figsize =( 8 , 6 ) )

    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

    _ = sns.heatmap(

        corr, 

        cmap = cmap,

        square=True, 

        cbar_kws={ 'shrink' : .9 }, 

        ax=ax, 

        annot = True, 

        annot_kws = { 'fontsize' : 12 }

    )



plot_correlation_map( df )
df.hist()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTXM0Y1smfhFyyGi0LXZPPSUCHWLgTy6tXAI2JE9ZNdSKsQH4LQ&usqp=CAU',width=400,height=400)