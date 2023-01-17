#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcT07i1_hYlNJQPvUFcc-vS_9IUGtbdJWwn6mZY2_YWOpvi2ypn1&usqp=CAU',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

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
nRowsRead = 1000 # specify 'None' if want to read whole file

df = pd.read_csv('../input/russian-corpus-of-biographical-texts/corpora.csv', delimiter=';', encoding = "ISO-8859-1", nrows = nRowsRead)

df.dataframeName = 'corpora.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')
df.tail()
fig,axes = plt.subplots(1,1,figsize=(20,5))

sns.heatmap(df.isna(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
# filling missing values with NA

df[['birth', 'death']] = df[['birth', 'death']].fillna('NA')
px.histogram(df, x='person', color='birth')
px.histogram(df, x='occupation', color='person')
fig = px.histogram(df, x='death', color='person')

fig.update_layout(showlegend=False)

fig.show()
fig = px.bar(df,

             y='occupation',

             x='person',

             orientation='h',

             color='death',

             title='Russian Corpora',

             opacity=0.8,

             color_discrete_sequence=px.colors.diverging.Armyrose,

             template='plotly_dark'

            )

fig.update_xaxes(range=[0,35])

fig.show()
fig = px.bar(df, x= "occupation", y= "person", color_discrete_sequence=['crimson'],)

fig.show()
# Scatter Matrix Plot

fig = px.scatter_matrix(df)

fig.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQcVxYDkcFNl4nW2wtVMlSPiXlPKq3VRz0TUOQOyyMpdlXkeUdc&usqp=CAU',width=400,height=400)