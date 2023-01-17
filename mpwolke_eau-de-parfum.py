#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSzhJUBt89c3mvgdezQ0PMMxgk-gwDgDlwSHBoGcKbK47JZw9cG',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

from plotly.offline import iplot

import plotly.express as px

import pandas_profiling as pp



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcS0aF_SCuH4jJ2ofN4x6h9C6939HOcHA7WdQvNoPtd5ac-11Wgd',width=400,height=400)
df = pd.read_excel('/kaggle/input/perfume-odor-dataset/perfume_data.xlsx')

df.head()
df.dtypes
sns.barplot(x=df['64541.7'].value_counts().index,y=df['64541.7'].value_counts())
#sample codes from Mikey_Mtk @motokinakamura https://www.kaggle.com/motokinakamura/treemap-with-plotly

fig = go.Figure(go.Treemap(

    labels = ["Eve","Cain", "Seth", "Enos", "Noam", "Abel", "Awan", "Enoch", "Azura"],

    parents = ["", "Eve", "Eve", "Seth", "Seth", "Eve", "Eve", "Awan", "Eve"]

))



fig.show()
plt.figure(figsize=(10,10))

plt.title('Eau de Parfum')

ax=sns.heatmap(df.corr(),

               linewidth=2.6,

               annot=True,

               center=1)
df["64543.1"].plot.hist()

plt.show()
df["64543.2"].plot.box()

plt.show()
df.plot(kind='scatter', x='64541.13', 

              y='64541.10', alpha=0.5, 

              color='darkblue', figsize = (6,4))



plt.title('Eau de Parfum')

plt.xlabel("64541.13")

plt.ylabel("64541.10")



plt.show()
df.plot(kind='scatter', x='64541.12', y='64528.5', 

              alpha=0.5, color='maroon', figsize = (6,4))



plt.title('Eau de Parfum')

plt.xlabel("64541.12")

plt.ylabel("64528.5")



plt.show()
trace1 = go.Box(

    y=df["64541.12"],

    name = '64541.12',

    marker = dict(color = 'rgb(0,145,119)')

)



trace2 = go.Box(

    y=df["64528.4"],

    name = '64528.4',

    marker = dict(color = 'rgb(5, 79, 174)')

)



data = [trace1, trace2]

layout = dict(autosize=False, width=700,height=500, title='Eau de Parfum', paper_bgcolor='rgb(243, 243, 243)', 

              plot_bgcolor='rgb(243, 243, 243)', margin=dict(l=40,r=30,b=80,t=100,))



fig = dict(data=data, layout=layout)

iplot(fig)
trace1 = go.Box(

    y=df["64528.4"],

    name = '64528.4',

    marker = dict(color = 'rgb(255,111,145)')

)



trace2 = go.Box(

    y=df["64541.4"],

    name = '64541.4',

    marker = dict(color = 'rgb(132,94,194)')

)



data = [trace1, trace2]

layout = dict(autosize=False, width=700,height=500, title='Hillshade 3pm and Noon', paper_bgcolor='rgb(243, 243, 243)', 

              plot_bgcolor='rgb(243, 243, 243)', margin=dict(l=40,r=30,b=80,t=100,))



fig = dict(data=data, layout=layout)

iplot(fig)
df = df.rename(columns={'64541.4':'seis', '64528.4': 'six'})
f,ax=plt.subplots(1,2,figsize=(15,7))

df.seis.plot.hist(ax=ax[0],bins=30,

                                                  edgecolor='black',color='crimson') 

                                       

ax[0].set_title('Eau de Parfum')

x1=list(range(-150,350,50))

ax[0].set_xticks(x1)



df.six.plot.hist(ax=ax[1],bins=30,

                                                    edgecolor='black',color='darkmagenta') 

                                                                                                        

ax[1].set_title('Eau de Cologne')

x2=list(range(0,1000,100))

ax[1].set_xticks(x2)



plt.show
parfum_types = df.iloc[:,14:-1].sum(axis=0)



plt.figure(figsize=(18,9))

sns.barplot(x=parfum_types.index, y=parfum_types.values, 

            palette="rocket")



plt.xticks(rotation= 75)

plt.ylabel('Total')

plt.title('Eau de Parfum',

          color = 'darkred',fontsize=12)



plt.show()
scent = df.iloc[:,10:14].sum(axis=0)



plt.figure(figsize=(7,5))

sns.barplot(x=scent.index, y=scent.values, 

            palette="Blues_d")



plt.xticks(rotation=90)

plt.title('Eau de Parfum',color = 'darkred',fontsize=12)

plt.ylabel('Parfum')



plt.show()
f,ax=plt.subplots(1,3,figsize=(21,7))

df.plot.scatter(ax=ax[0],x='64541.10', y='64541.13', 

                      alpha=0.5, color='purple')



ax[0].set_title('Eau de Parfum')

x1=list(range(1,8,1))

ax[0].set_ylabel("Parfum")

ax[0].set_xlabel("Eau de Parfum")

df.plot.scatter(ax=ax[1],x='63529.1', y='64528.1', 

                      alpha=0.5, color='purple')



ax[1].set_title('Parfum')

x2=list(range(1,8,1))

ax[1].set_ylabel("Eau de Parfum")

ax[1].set_xlabel("Parfum")

df.plot.scatter(ax=ax[2],x='63529.1', y='64528.5', 

                      alpha=0.5, color='purple')



ax[2].set_title('Eau de Cologne')

x2=list(range(1,8,1))

ax[2].set_ylabel("Cologne")

ax[2].set_xlabel("Eau de Cologne")



plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://www.pairfum.com/wp-content/uploads/2019/09/Eau-De-Parfum-Biggest-Single-Frustration-Woman-Test-600x600.jpg',width=400,height=400)