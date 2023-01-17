# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_wash_post = pd.read_csv('https://raw.githubusercontent.com/washingtonpost/data-police-shootings/master/fatal-police-shootings-data.csv')
df_wash_post['Shooting_Deaths'] = 1

df_wash_post['year'] = pd.DatetimeIndex(df_wash_post['date']).year

df_wash_post.head()
pd_stat=df_wash_post.groupby(['race'])['Shooting_Deaths'].sum()

pd_stat.plot(kind="bar")

plt.show()
df_wash_post["year_range"] = "Killings by Police 2015-2020"  # in order to have a single root node

fig = px.treemap(df_wash_post, path=['year_range', 'state', 'city'], values='Shooting_Deaths',

                 hover_data=['state'], color_continuous_scale='RdBu')

fig.show(rendering = "kaggle")
df_black = df_wash_post[df_wash_post['race'] == "B"]
df_black["year_range"] = "African American Killings by Police 2015-2020" 

fig = px.treemap(df_black, path=['year_range','state', 'city'], values='Shooting_Deaths',

                 hover_data=['state'], color_continuous_scale='RdBu')

fig.show(rendering = "kaggle")