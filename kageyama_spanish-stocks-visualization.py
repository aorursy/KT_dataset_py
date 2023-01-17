# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



os.chdir('../input')



# Any results you write to the current directory are saved as output.
# get all file name

filenames = [x for x in os.listdir() if x.endswith('.csv') and os.path.getsize(x) > 0]

print(filenames)
import plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



# make close value trace



r = lambda: random.randint(0,255)

traces = []



for filename in filenames:

    # random color create

    color = 'rgb({},{},{})'.format(str(r()),str(r()),str(r()))

    # get stock name 

    stock_name = filename.replace('.csv', '')

    # load csv

    df = pd.read_csv('../input/{}'.format(filename))

    # create line plot

    trace = go.Scatter(x=df.Date, y=df['Close'],name=stock_name,line=dict(color = color))

    traces.append(trace)



layout = py.graph_objs.Layout(

    title='Close Plot',

)

fig = py.graph_objs.Figure(data=traces, layout=layout)



py.offline.iplot(fig)
# make log close value trace



r = lambda: random.randint(0,255)

traces = []



for filename in filenames:

    # random color create

    color = 'rgb({},{},{})'.format(str(r()),str(r()),str(r()))

    # get stock name 

    stock_name = filename.replace('.csv', '')

    # load csv

    df = pd.read_csv('../input/{}'.format(filename))

    # create line plot

    trace = go.Scatter(x=df.Date, y=np.log(df['Close']),name=stock_name,line=dict(color = color))

    traces.append(trace)



layout = py.graph_objs.Layout(

    title='Log Plot',

)

fig = py.graph_objs.Figure(data=traces, layout=layout)



py.offline.iplot(fig)
# make price diff trace



r = lambda: random.randint(0,255)

traces = []



for filename in filenames:

    # random color create

    color = 'rgb({},{},{})'.format(str(r()),str(r()),str(r()))

    # get stock name 

    stock_name = filename.replace('.csv', '')

    # load csv

    df = pd.read_csv('../input/{}'.format(filename))

    # create line plot

    trace = go.Scatter(x=df.Date, y=df['Close'] - df['Open'],name=stock_name,line=dict(color = color))

    traces.append(trace)



layout = py.graph_objs.Layout(

    title='Price Diff Plot',

)

fig = py.graph_objs.Figure(data=traces, layout=layout)



py.offline.iplot(fig)