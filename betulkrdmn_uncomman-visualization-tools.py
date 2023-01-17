# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib_venn as venn

from math import pi

from pandas.tools.plotting import parallel_coordinates

import plotly.graph_objs as go

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import warnings

warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dictionary = {"column1":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,np.nan,np.nan,np.nan,np.nan],

              "column2":[1,2,3,4,np.nan,6,7,8,np.nan,10,np.nan,12,13,14,15,16,np.nan,18,np.nan,20],

              "column3":[1,2,3,4,np.nan,6,7,8,9,10,11,12,13,np.nan,15,16,17,18,np.nan,20]}







data_missingno=pd.DataFrame(dictionary)



import missingno as msno

msno.matrix(data_missingno)

plt.show()





data_missingno #value is null -> white not null -> black
msno.bar(data_missingno)

plt.show()
data=pd.read_csv('../input/Iris.csv')



data=data.drop(['Id'],axis=1) #Id feature is delete
data.head()
data.Species.unique()
plt.figure(figsize=(15,10)) #create 15x10 figure

parallel_coordinates(data,'Species',colormap=plt.get_cmap("Set2"))

plt.title("Iris data class visulalization according to features")

plt.xlabel("Features of data set")

plt.ylabel("cm")

plt.savefig('graph.png')

plt.show()

#usually use to for Social Network analysis



data1=data.iloc[:,0:4].corr()

#corr use for correlation between 0-1 

data1

import networkx as nx

#import library







links= data1.stack().reset_index() 

links.columns=['var1','var2','value']

#not table => value 1 value 2 and relationship e.g sepalWidthcm(var1) PetalLenghtCm(var2) -0.4(value)







threshold=0.5

#if correlation bigger than threshold connection established



links_filtered=links.loc[(links['value']>=threshold) & (links['var1']!=links['var2'])]





















df = data.loc[:,["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]

df1 = data.Species

x = dict(zip(df1.unique(),"rgb"))

row_colors = df1.map(x)

cg = sns.clustermap(df,row_colors=row_colors,figsize=(12, 12),metric="correlation")

plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(),rotation = 0,size =8)

plt.show()
trace1=go.Scatter(

    x=df.index,

   y=df.SepalLengthCm,

    xaxis='x2',

    yaxis='y2',

    mode="markers",

    name="income",

    marker=dict(color='rgba(0,112,20,0.8)'),

)

trace2=go.Histogram(

      x=df.SepalLengthCm,

    opacity=0.8,

    name='SepalLenghtCm',

    marker=dict(color='rgba(100,15,50,0.6)')

)



layout = go.Layout(

    xaxis2=dict(

        domain=[0.7, 1],

        anchor='y2',        

    ),

   yaxis2=dict(

        domain=[0.6,0.95],

           anchor='x2',

           ),

    title='Sepal Lenght Histogram and Scatter  Plot'

)





fig=go.Figure(data=data,layout=layout)

iplot(fig)













data=pd.read_csv('../input/Iris.csv')



iris_setosa = data[data.Species == "Iris-setosa"]





iris_virginica = data[data.Species == "Iris-virginica"]



trace1 = go.Scatter3d(

    x=iris_setosa.SepalLengthCm,

    y=iris_setosa.SepalWidthCm,

    z=iris_setosa.PetalLengthCm,

    mode='markers',

    name = "iris_setosa",

    marker=dict(

        color='rgb(21, 100, 100)',

        size=12,

        line=dict(

            color='rgb(255, 255, 255)',

            width=0.1

        )

    )

)

trace2=go.Scatter3d(

    x=iris_virginica.SepalLengthCm,

    y=iris_virginica.SepalWidthCm,

    z=iris_virginica.PetalLengthCm,

    mode='markers',

    name = "iris_virginica",

    marker=dict(

          color='rgb(54, 170, 127)',

        size=12,

        line=dict(

            color='rgb(205, 205, 205)',

            width=0.1

        )

    )

)

data=[trace1,trace2]

layout=go.Layout(

       title='3D iris-virginica and iris-setosa',

       margin=dict(l=0,r=0,t=0,b=0)   

   

)

fig=go.Figure(data=data,layout=layout)

iplot(fig)







# import data again

data = pd.read_csv('../input/Iris.csv')

# data of iris setosa

iris_setosa = data[data.Species == "Iris-setosa"]

# # data of iris virginica

iris_virginica = data[data.Species == "Iris-virginica"]



# trace1 =  iris setosa

trace1 = go.Scatter3d(

    x=iris_setosa.SepalLengthCm,

    y=iris_setosa.SepalWidthCm,

    z=iris_setosa.PetalLengthCm,

    mode='markers',

    name = "iris_setosa",

    marker=dict(

        color='rgb(217, 100, 100)',

        size=12,

        line=dict(

            color='rgb(255, 255, 255)',

            width=0.1

        )

    )

)

# trace2 =  iris virginica

trace2 = go.Scatter3d(

    x=iris_virginica.SepalLengthCm,

    y=iris_virginica.SepalWidthCm,

    z=iris_virginica.PetalLengthCm,

    mode='markers',

    name = "iris_virginica",

    marker=dict(

        color='rgb(54, 170, 127)',

        size=12,

        line=dict(

            color='rgb(204, 204, 204)',

            width=0.1

        )

    )

)

data = [trace1, trace2]

layout = go.Layout(

    title = ' 3D iris_setosa and iris_virginica',

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0

    )

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)