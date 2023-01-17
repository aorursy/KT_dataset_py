#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRuS7jsGSFkyudc_hnJBIi5KYdw11kPxRxOWV655fMkfCbdm7WD&usqp=CAU',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px

import seaborn



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_excel('/kaggle/input/covid19-chinesesocialmedia-hotspots/Douyin_2020Coron.xlsx')

df.head()
df = df.rename(columns={'Unnamed: 0':'unnamed', 'Coron-Related(1 yes, 0 not)': 'related'})
fig,axes = plt.subplots(1,1,figsize=(20,5))

sns.heatmap(df.isna(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
bboxtoanchor=(1.1, 1.05)

seaborn.set(rc={'axes.facecolor':'03c6fc', 'figure.facecolor':'03c6fc'})

df.plot.area(y=['unnamed','searchCount','rank'],alpha=0.4,figsize=(12, 6));
seaborn.set(rc={'axes.facecolor':'purple', 'figure.facecolor':'purple'})

sns.countplot(df["rank"])

plt.xticks(rotation=90)

plt.show()
seaborn.set(rc={'axes.facecolor':'purple', 'figure.facecolor':'purple'})

fig=sns.lmplot(x="searchCount", y="rank",data=df)
#Code from Prashant Banerjee @Prashant111

labels = df['rank'].value_counts().index

size = df['rank'].value_counts()

colors=['cyan','crimson']

plt.pie(size, labels = labels, colors = colors, shadow = True, autopct='%1.1f%%',startangle = 90)

plt.title('Douyin China Social Media', fontsize = 20)

plt.legend()

plt.show()
fig = px.density_contour(df, x="searchCount", y="rank", color_discrete_sequence=['purple'])

fig.show()
fig = px.density_contour(df, x="unnamed", y="rank", color_discrete_sequence=['crimson'])

fig.show()
cnt_srs = df['title'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Blues',

        reversescale = True

    ),

)



layout = dict(

    title='Douyin China Social Media',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="title")
seaborn.set(rc={'axes.facecolor':'03c6fc', 'figure.facecolor':'03c6fc'})

df["searchCount"].plot.hist()

plt.show()
seaborn.set(rc={'axes.facecolor':'03c6fc', 'figure.facecolor':'03c6fc'})

fig=plt.gcf()

fig.set_size_inches(10,7)

fig=sns.violinplot(x='unnamed',y='rank',data=df)
seaborn.set(rc={'axes.facecolor':'03c6fc', 'figure.facecolor':'03c6fc'})

df.corr()

plt.figure(figsize=(10,4))

sns.heatmap(df.corr(),annot=True,cmap='YlOrRd_r')

plt.show()
seaborn.set(rc={'axes.facecolor':'03c6fc', 'figure.facecolor':'03c6fc'})

corr = df.corr(method='pearson')

sns.heatmap(corr)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcShKmva47KbhmOxmcwyvbN8g0dbeRXPL_NiFmYeI5YhH8nTBe6O&usqp=CAU',width=400,height=400)