# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

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
df = pd.read_csv('../input/hackathon/task_2-owid_covid_data-21_June_2020.csv')

df.head()
fig = px.parallel_categories(df, color="extreme_poverty", color_continuous_scale=px.colors.sequential.Viridis)

fig.show()
fig = px.parallel_categories(df, color="gdp_per_capita", color_continuous_scale=px.colors.sequential.OrRd)

fig.show()
fig = px.parallel_categories(df, color="handwashing_facilities", color_continuous_scale=px.colors.sequential.RdBu)

fig.show()