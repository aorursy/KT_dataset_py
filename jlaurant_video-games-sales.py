# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/Video_Games_Sales_as_at_22_Dec_2016.csv')
df.head()
df.shape
df.Platform.value_counts().head(15)
df.Publisher.value_counts().head(10)
import cufflinks as cf

import plotly

plotly.offline.init_notebook_mode()

cf.go_offline()

import warnings

warnings.filterwarnings('ignore')
# Top publishers in terms of global sales

data = df.groupby('Publisher').sum()

fig = data.sort_values('Global_Sales', ascending=False)[:10]

fig = fig.pivot_table(index=['Publisher'], values=['NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales'], aggfunc=np.sum)

fig = fig.sort_values('Global_Sales', ascending=True)

fig = fig.drop('Global_Sales', axis=1)

fig = fig.iplot(kind='barh', barmode='stack' , asFigure=True) #For plotting with Plotly

fig.layout.margin.l = 200 #left margin distance

fig.layout.xaxis.title='Sales in Million Units'# For setting x label

fig.layout.title = "Top 10 global publisher sales" # For setting the graph title

plotly.offline.iplot(fig) # Show the graph
pub=['Nintendo','Activision','Sony Computer Entertainment','Electronic Arts']

data_pubs= df[df.Publisher.isin(pub)]
# Compare yearly performance of three publishers

fig = (data_pubs.pivot_table(

        index=['Year_of_Release'], values=['Global_Sales'], columns=['Publisher'], 

        aggfunc=np.sum, dropna=False,

        )['Global_Sales'].iplot(

        subplots=True, subplot_titles=True, asFigure=True, fill=True,title='Sales per year by Publisher'))

fig.layout.height= 800

fig.layout.showlegend=False 

plotly.offline.iplot(fig)
# Compare release with other competitors: more release, more sales

top_publish=df['Publisher'].value_counts().sort_values(ascending=False)[:10]

top_publish=df[df['Publisher'].isin(top_publish.index)].groupby(

    ['Publisher','Year_of_Release'])['Name'].count().reset_index()

top_publish=top_publish.pivot('Year_of_Release','Publisher','Name')

top_publish[top_publish.columns[:-1]].iplot(

    kind='heatmap',colorscale='RdYlGn',title='Publisher Games Releases By Years')