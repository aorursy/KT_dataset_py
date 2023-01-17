#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQotwgH8SiQyKYiEbYE6NrtH8tQzcP1f6b1yoB9j-BtSB36HZF1&usqp=CAU',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px

from plotly.offline import iplot



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_excel('/kaggle/input//crime-in-russia-20032020/crime.xlsx')

df.head()
fig,axes = plt.subplots(1,1,figsize=(20,5))

sns.heatmap(df.isna(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
df.dropna(how = 'all',inplace = True)

df.drop(['Unnamed: 3', 'Unnamed: 4', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 9', 'Unnamed: 10', 'Unnamed: 12', 'Unnamed: 13', 'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 18', 'Unnamed: 19', 'Unnamed: 21', 'Unnamed: 22', 'Unnamed: 24', 'Unnamed: 25', 'Unnamed: 27', 'Unnamed: 28', 'Unnamed: 30', 'Unnamed: 31', 'Unnamed: 33', 'Unnamed: 34', 'Unnamed: 36', 'Unnamed: 37', 'Unnamed: 39', 'Unnamed: 40', 'Unnamed: 42', 'Unnamed: 43', 'Unnamed: 45', 'Unnamed: 46', 'Unnamed: 48', 'Unnamed: 49', 'Unnamed: 51', 'Unnamed: 52'],axis=1,inplace = True)

df.shape
plot_data = df.groupby(['год, месяц'], as_index=False).мошенничество.sum()



fig = px.line(plot_data, x='год, месяц', y='мошенничество')

fig.show()
plot_data = df.groupby(['кража траспортных средств'], as_index=False).экологические.sum()



fig = px.line(plot_data, x='кража траспортных средств', y='экологические')

fig.show()
cnt_srs = df['экстремистской направленности'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Purples',

        reversescale = True

    ),

)



layout = dict(

    title='экстремистской направленности',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="экстремистской направленности")
cnt_srs = df['убийство и покушение на убийство'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Greens',

        reversescale = True

    ),

)



layout = dict(

    title='убийство и покушение на убийство',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="")
cnt_srs = df['незаконное хранение и оборот оружия'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Reds',

        reversescale = True

    ),

)



layout = dict(

    title='',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="незаконное хранение и оборот оружия")
# Grouping it by age and province

plot_data = df.groupby(['год, месяц', 'кража траспортных средств'], as_index=False).мошенничество.sum()



fig = px.bar(plot_data, x='год, месяц', y='мошенничество', color='кража траспортных средств')

fig.show() 
dfcorr=df.corr()

dfcorr
plt.figure(figsize=(10,4))

sns.heatmap(df.corr(),annot=False,cmap='vlag')

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSY009Fg6PfnXMT7FMQK1KK8iXnezMMBxRESzF-yxIq3tC1CPCR&usqp=CAU',width=400,height=400)