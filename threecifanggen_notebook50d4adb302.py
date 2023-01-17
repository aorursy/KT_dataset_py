# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.import warnings

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

import pandas as pd

import plotly

import seaborn as sb

import bokeh

import scipy as sp

import statsmodels as sm

from IPython import display

import datetime

import ggplot

import statsmodels

from bokeh.io import output_notebook

output_notebook()

from bokeh.charts import Bar, show, Scatter

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go

init_notebook_mode()

import functools as ft

import itertools as it

from ipywidgets import *
trace_1 = go.Scatter(

    x = [1,2,3],

    y = [2,3,4]

)

iplot(go.Figure(data=[trace_1]))
@interact(x=(1,2))

def f(x):

    return x **2