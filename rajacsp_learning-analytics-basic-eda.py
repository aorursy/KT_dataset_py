# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import re

import matplotlib.pyplot as plt



import plotly.graph_objects as go

import plotly.express  as px
FILEPATH = '/kaggle/input/learning-analytics/LA-Links-2020-Sep-v2.csv'
df = pd.read_csv(FILEPATH)
df.head()
df = df.drop(['_id', 'comment'], axis = 1)
df = df.reset_index()
df.head()
# Which user collected the most?



temp_df = pd.DataFrame(df['collected_by'].value_counts().head(10)).reset_index()



temp_df
fig = go.Figure(data=[go.Pie(labels=temp_df['index'],

                             values=temp_df['collected_by'],

                             hole=.7,

                             title = '% links by User',

                             marker_colors = px.colors.sequential.Blues_r,

                            )

                     

                     ])

fig.update_layout(title = '% links by User')

fig.show()