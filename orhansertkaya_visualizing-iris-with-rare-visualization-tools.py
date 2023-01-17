# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib_venn as venn
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
iris = pd.read_csv('../input/Iris.csv')
iris.info()
iris = iris.drop(['Id'],axis=1)
new_iris = iris.iloc[:,:3]
new_iris.SepalLengthCm[np.arange(1,150,10)] = np.nan
new_iris.PetalLengthCm[np.arange(25,120,7)] = np.nan
new_iris
# import missingno library
import missingno as msno

msno.matrix(new_iris)
plt.show()
# Make the plot
plt.figure(figsize=(15,10))
parallel_coordinates(iris, 'Species', colormap=plt.get_cmap("Set2"))
plt.title("Iris data class visualization according to features (setosa, versicolor, virginica)")
plt.xlabel("Features of data set")
plt.ylabel("cm")
plt.savefig('graph.png')
plt.show()
# Display positive and negative correlation between columns
iris.corr()
#sorts all correlations with ascending sort.
iris.corr().unstack().sort_values().drop_duplicates()
iris.corr().stack().reset_index()
# import networkx library
import networkx as nx

# Transform it in a links data frame (3 columns only):
links = iris.corr().stack().reset_index()
links.columns = ['var1', 'var2','value']

# correlation
threshold = -1

# Keep only correlation over a threshold and remove self correlation (cor(A,A)=1)
links_filtered=links.loc[ (links['value'] >= threshold ) & (links['var1'] != links['var2']) ]
 
# Build your graph
G=nx.from_pandas_edgelist(links_filtered, 'var1', 'var2')
 
# Plot the network
nx.draw_circular(G, with_labels=True, node_color='green', node_size=1000, edge_color='cyan', linewidths=3, font_size=12)

# venn2
from matplotlib_venn import venn2
data_1 = len(iris.SepalLengthCm)
data_2 = len(iris.SepalWidthCm)
data_3 = len(iris[(iris.SepalLengthCm==iris.SepalWidthCm)]) # =0

# First way to call the 2 group Venn diagram
venn2(subsets = (data_1, data_2, data_3), set_labels = ('SepalLengthCm', 'SepalWidthCm'))
plt.show()
# donut plot
feature_names = "sepal_length","sepal_width","petal_length","petal_width"
feature_size = [len(iris.SepalLengthCm),len(iris.SepalWidthCm),len(iris.PetalLengthCm),len(iris.PetalWidthCm)]
# create a circle for the center of plot
circle = plt.Circle((0,0),0.5,color = "white") #(0,0) coordinate
plt.pie(feature_size, labels = feature_names, colors = ["black","green","blue","cyan"] )
p = plt.gcf()
p.gca().add_artist(circle)
plt.title("Number of Each Features")
plt.show()
# spider graph
categories = list(iris)[:4]
N = len(categories)
angles = [ n / float(N)*2*pi for n in range(N)]
angles = angles + angles[:1]
plt.figure(figsize = (10,10))
ax = plt.subplot(111,polar = True)
ax.set_theta_offset(pi/2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1],categories)
ax.set_rlabel_position(0)
plt.yticks([0,2,4,6],["0","2","4","6"],color= "red", size = 7)
plt.ylim(0,6)

values = iris.loc[0].drop("Species").values.flatten().tolist()
values = values + values[:1]
ax.plot(angles,values,linewidth = 1,linestyle="solid",label ="setosa" )
ax.fill(angles,values,"b",alpha=0.1)

values = iris.loc[1].drop("Species").values.flatten().tolist()
values = values + values[:1]
ax.plot(angles,values,linewidth = 1,linestyle="solid",label ="versicolor" )
ax.fill(angles,values,"orange",alpha=0.1)
plt.legend(loc = "upper left",bbox_to_anchor = (0.1,0.1))
plt.show()
# trace1 is line plot
# go: graph object
trace1 = go.Scatter(
    x=iris.index,
    y=iris.SepalLengthCm,
    mode = "markers",
    xaxis='x2',
    yaxis='y2',
    name = "SepalLengthCm",
    marker = dict(color = 'rgba(76, 120, 213, 0.8)'),
)

# trace2 is histogram
trace2 = go.Histogram(
    x=iris.SepalLengthCm,
    opacity=0.75,
    name = "Sepal Length(Cm)",
    marker=dict(color='rgba(120, 5, 125, 0.6)'))

# add trace1 and trace2
data = [trace1, trace2]
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
fig = go.Figure(data=data, layout=layout)
iplot(fig)
# import data again
iris = pd.read_csv('../input/Iris.csv')
# data of iris setosa
iris_setosa = iris[iris.Species == "Iris-setosa"]
# data of iris virginica
iris_virginica = iris[iris.Species == "Iris-virginica"]
# data of iris virginica
iris_versicolor = iris[iris.Species == "Iris-versicolor"]

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
            color='rgb(0, 0, 0)',
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
            color='rgb(0, 0, 0)',
            width=0.1
        )
    )
)
# trace3 =  iris versicolor
trace3 = go.Scatter3d(
    x=iris_versicolor.SepalLengthCm,
    y=iris_versicolor.SepalWidthCm,
    z=iris_versicolor.PetalLengthCm,
    mode='markers',
    name = "iris_setosa",
    marker=dict(
        color='rgb(100, 150, 145)',
        size=12,
        line=dict(
            color='rgb(0, 0, 0)',
            width=0.1
        )
    )
)
data = [trace1, trace2, trace3]
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