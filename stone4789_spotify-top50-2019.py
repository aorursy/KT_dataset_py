# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import plotly.graph_objs as go

import plotly.express as px

file='/kaggle/input/top50spotify2019/top50.csv'

df=pd.read_csv(file,encoding='ISO-8859-1',header=0)

df.rename(columns={'Artist.Name':'Artist','Beats.Per.Minute': 'BPM', 'Track.Name': 'Title', 'Length.':'Length', 'Acousticness..':'Acousticness', 'Valence.':'Valence', 'Speechiness.':'Speechiness'}, inplace=True)

df.head()
#Graph BPM

fig = px.bar(df, x='Artist', y='BPM', hover_data=['Genre','Title'])

fig.show()
fig = px.scatter(df, x='BPM', y='Popularity', color='Genre', size='Length')

fig.show()
df.sort_values(by='Popularity', ascending=False).head(10)