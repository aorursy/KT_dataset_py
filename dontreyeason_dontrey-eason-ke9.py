# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pylab as plt

import plotly

import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

plt.style.use('ggplot')

color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]



# Format the data

df = pd.read_csv('../input/lanl-leaderboard/LANL-Earthquake-Prediction-publicleaderboard_05_31_2019.csv')

df['SubmissionDate'] = pd.to_datetime(df['SubmissionDate'])

df = df.set_index(['TeamName','SubmissionDate'])['Score'].unstack(-1).T

df.columns = [name for name in df.columns]



# Any results you write to the current directory are saved as output.