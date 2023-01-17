# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/coronavirus-source-data-covid19-daily-reports/total_deaths.csv")
df.head().style.background_gradient(cmap='copper')
df.dtypes
# Eksik veri sayıları ve veri setindeki oranları 

plt.figure(figsize=(8,8))

sns.heatmap(pd.isnull(df.T), cbar=False)



pd.concat([df.isnull().sum(), 100 * df.isnull().sum()/len(df)], 

              axis=1).rename(columns={0:'Missing Records', 1:'Percentage (%)'})
df.dropna(how = 'all',inplace = True)

df.drop(['Afghanistan','Albania','Algeria', 'United Arab Emirates', 'Vatican'],axis=1,inplace = True)

df.shape
sns.distplot(df["China"].apply(lambda x: x**4))

plt.show()
cnt_srs = df['China'].value_counts().head()

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

    title='China',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="China")
fig = px.histogram(df[df.China.notna()],x="China",marginal="box",nbins=10)

fig.update_layout(

    title = "China",

    xaxis_title="China",

    yaxis_title="Worldwide",

    barmode="group",

    bargap=0.1,

    xaxis = dict(

        tickmode = 'linear',

        tick0 = 0,

        dtick = 10),

    font=dict(

        family="Arial, monospace",

        size=15,

        color="#7f7f7f"

    )

    )

py.iplot(fig)
sns.countplot(df["China"])

plt.xticks(rotation=90)

plt.show()
cat = []

num = []

for col in df.columns:

    if df[col].dtype=='O':

        cat.append(col)

    else:

        num.append(col)  

        

        

num 
plt.style.use('dark_background')

for col in df[num].drop(['China'],axis=1):

    plt.figure(figsize=(8,5))

    plt.plot(df[col].value_counts(),color='Red')

    plt.xlabel(col)

    plt.ylabel('China')

    plt.tight_layout()

    plt.show()