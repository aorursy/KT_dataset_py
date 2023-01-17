# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Plotting Libraries

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="ticks", color_codes=True)

%matplotlib inline



from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly as py

import plotly.graph_objects as go

from plotly.subplots import make_subplots



init_notebook_mode(connected=True)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/fifa19/data.csv')

df.head(5)
df.columns
df_foot = df[['ID', 'Name', 'Age', 'Nationality', 'Preferred Foot', 'Weak Foot']]

df_foot.head()
df_foot['Preferred Foot'].unique().tolist()
df_foot_nopref = df_foot.loc[df_foot['Preferred Foot'].apply(lambda row: str(row).lower() not in ['right', 'left'])]

print('Total number of players with no Preferred or Weak Foot: {}'.format(df_foot_nopref.shape[0]))

df_foot_nopref
s = df_foot['Preferred Foot'].value_counts(dropna=False)

pd.DataFrame(s)
import plotly.graph_objects as go



x, y = s.index.tolist(), s.to_list()

x = [val if (repr(val)!='nan') else 'NA' for val in x] # this converts nan into 'NA'

total = df_foot.shape[0]



# Use textposition='auto' for direct text

fig = go.Figure(data=[go.Bar(

            x=x, y=y,

            text=y,

            textposition='auto',

            width=[0.3, 0.3, 0.3], 

            marker_color=['#5ab4ac', '#fc8d59', '#91cf60']

        )])



fig.update_layout(title_text='Total Number of Right-Foot vs. Left-Foot Players')

fig.show()
x
import plotly.graph_objs as go



labels = x.copy()

values = y.copy() 

#colors = ['aliceblue',  'aqua', 'aquamarine', 'darkturquoise']

colors = [ '#fc8d59', '#ffffbf', '#91cf60', ] # colors for: ['Right', 'Left', 'NA']



trace = go.Pie(labels=labels, values=values,

               hoverinfo='label+percent', textinfo='value', 

               textfont=dict(size=20),

               marker=dict(colors=colors, 

                           line=dict(color='rgb(100,100,100)', 

                                     width=1)

                          )

              )



fw=go.FigureWidget(data=[trace])

fw