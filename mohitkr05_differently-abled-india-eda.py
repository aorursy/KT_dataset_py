# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# USING SEABORN LIBRARY MOSTLY

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # this is used for the plot the graph 

import seaborn as sns # used for plot interactive graph.

import warnings

warnings.filterwarnings('ignore')

from pylab import rcParams

# figure size in inches

%matplotlib inline

import plotly.graph_objs as go

import plotly

plotly.offline.init_notebook_mode()





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/disabled-community-dataset/disabled_community_dataset.csv')
df.head()
df.info()
df.isna().sum()
df["state"].unique()
df1 = df[df["state"] != "india"]
df1.head()
sns.set_color_codes("pastel")

dx = df1.sort_values(["number_disabled"], ascending=False)

fig_dims = (16, 8)

fig, ax = plt.subplots(figsize=fig_dims)

sns.barplot(x = 'number_disabled', y = 'state', data = dx,

            label = 'Number of differently abled', color = 'b', edgecolor = 'w')
sns.set_color_codes("pastel")

x = df1.sort_values(["workforce_rate_disabled"], ascending=False)

fig_dims = (16, 8)

fig, ax = plt.subplots(figsize=fig_dims)

sns.barplot(x = 'workforce_rate_disabled', y = 'state', data = x,

            label = 'Literacy rate among differently abled', color = 'b', edgecolor = 'w')
sns.set_color_codes("pastel")

x = df1.sort_values(["literacy_rate_disabled"], ascending=False)

fig_dims = (16, 8)

fig, ax = plt.subplots(figsize=fig_dims)

sns.barplot(x = 'literacy_rate_disabled', y = 'state', data = x,

            label = 'Literacy rate among differently abled', color = 'b', edgecolor = 'w')