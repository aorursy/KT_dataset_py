# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

import plotly.figure_factory as ff

py.init_notebook_mode(connected=True)

import plotly.graph_objects as go

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
df = pd.read_excel('/kaggle/input/amofah-health-data/Amofah Health Data.xlsx')

df.head()
fig = px.bar(df, 

             x='Record value', y='Data Source Name', color_discrete_sequence=['darkgreen'],

             title='Amofah Health Data', text='Source Version')

fig.show()
fig = px.bar(df, 

             x='Record value', y='Record Unit', color_discrete_sequence=['crimson'],

             title='Amofah Health Data', text='Record type')

fig.show()
fig = px.line(df, x="Start Date", y="Record type", color_discrete_sequence=['darkseagreen'], 

              title="Amofah Health Data")

fig.show()