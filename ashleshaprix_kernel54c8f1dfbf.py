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
import numpy as np 

import pandas as pd



import seaborn as sns



import matplotlib.pyplot as plt

plt.style.use('ggplot')

import plotly.offline as py

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode

init_notebook_mode(connected=False)
df=pd.read_csv("/kaggle/input/vegetablepricetomato/Price of Tomato Karnataka(2016-2018).csv")
print("dataset contains {} rows and {} columns".format(df.shape[0],df.shape[1]))

df.head()
df.info()
df.describe()
np.sum(df.isna())
df.corr()
df['Arrival Date'].fillna(0,inplace = True)
df['Arrivals (Tonnes)'].fillna(0,inplace = True)

df['Variety'].fillna(0,inplace = True)
np.sum(df.isna())
plt.figure(figsize=(10,7))

chains=df['Market'].value_counts()[:30]

sns.barplot(x=chains,y=chains.index,palette='deep')

plt.title("Most famous Market for Tomato Seller")

x=df['Variety'].value_counts()

colors = ['#FEBFB3', '#E1396C']



trace=go.Pie(labels=x.index,values=x,textinfo="value",

            marker=dict(colors=colors, 

                           line=dict(color='#000000', width=2)))

layout=go.Layout(title="Different Variety's of Tomato",width=500,height=500)

fig=go.Figure(data=[trace],layout=layout)

py.iplot(fig, filename='pie_chart_subplots')

    

    