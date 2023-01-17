# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

import gc

import warnings

warnings.filterwarnings('ignore')

pd.set_option('max_columns', 150)

pd.set_option('max_rows', 150)
import matplotlib.pyplot as plt

from matplotlib import rcParams

import seaborn as sns

from scipy import stats

#To plot figs on jupyter

%matplotlib inline

# figure size in inches

rcParams['figure.figsize'] = 14,6



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls


# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
PATH =   r"/kaggle/input/rawdata/RAW-METRICS-2014-2015/2015/2015-CW-12/"

df = pd.read_csv(PATH+"lphost10_global_unix-global_prod-appserver_2015.03.22_00.00.01.csv")
df.columns
df.head()
from plotly import tools, subplots

import plotly.offline as py

py.init_notebook_mode(connected = True)



def line_plot_check_nan(df1, df2, x, y, title, width, height):

    

    trace1 = go.Scatter(

        x = df1[x],

        y = df1[y],

        mode='lines',

        name='with_nans',

        marker = dict(

            color = '#1E90FF', 

        ), 

    )

    

    df3 = df2.dropna()

    trace2 = go.Scatter(

        x = df3[x],

        y = df3[y],

        mode='markers',

        name='no_nans',

        marker = dict(

            color = 'red', 

        ), 

    )

    

    layout = go.Layout(

        title = go.layout.Title(

            text = title,

            x = 0.5

        ),

        font = dict(size = 14),

        width = width,

        height = height,

    )

    

    data = [trace1, trace2]

    fig = go.Figure(data = data, layout = layout)

    py.iplot(fig, filename = 'line_plot')
line_plot_check_nan(df,df, 'Date', "\\Swap\\used"," Swap used by date", 1400, 600)
line_plot_check_nan(df,df,  'Date', "\\Swap\\free"," Swap used by date", 1400, 600)