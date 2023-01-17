# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



data = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
from plotly import tools

import plotly.offline as pyo

from plotly.offline import init_notebook_mode

import plotly.figure_factory as ff

import plotly.graph_objs as go

init_notebook_mode(connected=True)

columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']

plot_y = []

names = []

for i in columns:

    for x in data[i].unique():

        plot_y.append(data[data[i]==x].tenure.sum())

        names.append("{}and{}".format(i,x))

trace = [go.Bar(x=names,

               y=plot_y)]

pyo.iplot(trace)