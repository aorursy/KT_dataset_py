import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib_venn as venn

from math import pi

from pandas.plotting import parallel_coordinates

import plotly.graph_objs as go



from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import warnings

warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))
data=pd.read_csv('../input/iris/Iris.csv')

data=data.drop(['Id'], axis=1)

                

plt.figure(figsize=(15,10))

                

parallel_coordinates(data,'Species',colormap=plt.get_cmap("Set1"))                

plt.title('visualization according to species')

plt.xlabel("Features of data set")

plt.ylabel("cm")

plt.savefig('graph.png')

plt.show()

                

                
import missingno as msno



dictionary={"colum1":[1,3,np.nan,10,20,np.nan],

           "colum2":[5,32,5,54,np.nan,np.nan],

           "colum3":[1,2,5,6,7,2]}



data_missingo=pd.DataFrame(dictionary)

msno.matrix(data_missingo)

plt.show()
msno.bar(data_missingo)
corr = data.iloc[:,0:4].corr()

corr
import networkx as nx

import pandas as np

import numpy as np

link=corr.stack().reset_index()

link.columns = ['var1', 'var2','value']

threshold=-1

links_filtered=link.loc[(link['value']>threshold) & (link['var1']!=link['var2'])]

G=nx.from_pandas_edgelist(links_filtered,'var1','var2')

nx.draw_circular(G, with_labels=True, node_color='orange', node_size=300, edge_color='red', linewidths=1, font_size=10)



# venn2

from matplotlib_venn import venn2

sepal_length = data.iloc[:,0]

sepal_width = data.iloc[:,1]

petal_length = data.iloc[:,2]

petal_width = data.iloc[:,3]

# First way to call the 2 group Venn diagram

venn2(subsets = (len(sepal_length)-15, len(sepal_width)-15, 15), set_labels = ('sepal_length', 'sepal_width'))

plt.show()
data.PetalWidthCm
data.index
trace1=go.Scatter(x=data.index,

                  y=data.SepalLengthCm,

                  mode='markers',

                  xaxis='x2',

                  yaxis='y2',

                  name='SepalLengthCm',

                  marker = dict(color = 'rgba(0, 112, 20, 0.8)'),

)

trace2=go.Histogram(

                    x=data.SepalLengthCm,

                    opacity=0.8,

                    name='SepalLengthCm',

                    marker=dict(color='rgba(60,125,62,0.8)'))

yeni_data=[trace1,trace2]



layout = go.Layout(

    xaxis2=dict(

        domain=[0.7, 1],

        anchor='y2',        

    ),

    yaxis2=dict(

        domain=[0.6, 0.95],

        anchor='x2',

    ),

    title = ' Sepal Length(Cm) Histogram and Scatter Plot'

)



fig = go.Figure(data=yeni_data, layout=layout)

iplot(fig)

data=pd.read_csv('../input/iris/Iris.csv')

data
data=pd.read_csv('../input/iris/Iris.csv')



data1=data[data.Species== 'Iris-setosa']

data2=data[data.Species=='Iris-versicolor']



trace1=go.Scatter3d(x=data1.SepalLengthCm,

                    y=data1.SepalWidthCm,

                    z=data1.PetalWidthCm,

                    mode='markers',

                    name='setosa_featurs',                   

                    marker=dict(color='rgba(200,200,200,0.6)',size=12,line=dict(color='rgb(10,20,30)', width=0.2)))



trace2=go.Scatter3d(x=data2.SepalLengthCm,

                   y=data2.SepalWidthCm,

                   z=data2.PetalWidthCm,

                   mode='markers',

                   name='versicolor_feature',

                    marker=dict(

        color='rgb(217, 100, 100)',

        size=12,

        line=dict(

            color='rgb(255, 255, 255)',

            width=0.1)))





yeni_data=[trace1,trace2]



layout=go.Layout(title='3D iris and versicolor specious drawing',margin=dict(l=0,r=0,b=0,t=0))



fig=go.Figure(data=yeni_data,layout=layout)

iplot(fig)