# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# import graph objects as "go"

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import pandas as pd

model = pd.read_csv("../input/complete-educational-stats-unesco/EDULIT_DS_30112019025843883.csv")
print('This dataset has ' + str(model.shape[0]) + ' rows, and ' + str(model.shape[1]) + ' columns')
print(pd.isnull(model).sum())
#model = model.fillna({"Value": 0})

model = model.dropna(subset=['Value'], how='all')
model.describe()
sns.distplot(model.Time.dropna(), kde=False, bins = 39);