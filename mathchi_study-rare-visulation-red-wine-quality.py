# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib_venn as venn

from math import pi

from pandas.plotting import parallel_coordinates

import plotly.graph_objs as go

import plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import warnings

warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# NaN value leri gorebilmek icin rastgele nan value ler verelim



# Define dictionary

dictionary = {"column1":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],

              "column2":[1,2,3,4,np.nan,6,7,8,np.nan,10,np.nan,12,13,14,15,16,np.nan,18,np.nan,20],

              "column3":[1,2,3,4,np.nan,6,7,8,9,10,11,12,13,np.nan,15,16,17,18,np.nan,20]}





# Create data frame from dictionary

data_missingno = pd.DataFrame(dictionary) 
data_missingno.head(10)
# Hazirladigimiz dataframe gore gorsel olarak bakalim





# import missingno library



import missingno as msno



msno.matrix(data_missingno)

plt.show()
# missingno bar plot

msno.bar(data_missingno)

plt.show()
# load iris data

data = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

data.head()
data.rename(columns = {'fixed acidity': 'fixed_acidity', 'volatile acidity': 'volatile_acidity', 'citric acid': 'citric_acid', 'residual sugar': 'residual_sugar',

       'free sulfur dioxide': 'free_sulfur_dioxide', 'total sulfur dioxide': 'total_sulfur_dioxide'}, inplace=True)

data.columns
data.info()
# Make the plot

plt.figure(figsize=(15,10))

parallel_coordinates(data, 'quality', colormap=plt.get_cmap("Set1"))

plt.title("Red Wine data class visualization according to quality (3,4,5,6,7,8)")

plt.xlabel("Ingredients of data set")

plt.ylabel("level")

plt.savefig('graph.png')

plt.show()
# Calculate the correlation between individuals.

corr = data.iloc[:,0:10].corr()

corr
# import networkx library

import networkx as nx



# Transform it in a links data frame (3 columns only):

links = corr.stack().reset_index()

links.columns = ['var1', 'var2','value']     # sadece 3 tane sutun aliyoruz cunku 4 ozellikten kendisi haric olani almaliyiz yani 3 adet

links.head(10)
# correlation

threshold = -1           # Simdi bu esik degerine gore aradaki bagi gosteren bir grafik fizelim



# Keep only correlation over a threshold and remove self correlation (cor(A,A)=1)

links_filtered=links.loc[ (links['value'] >= threshold ) & (links['var1'] != links['var2']) ]

 

# Build your graph

G=nx.from_pandas_edgelist(links_filtered, 'var1', 'var2')

 

# Plot the network

nx.draw_circular(G, with_labels=True, node_color='orange', node_size=300, edge_color='red', linewidths=1, font_size=15)
data.head()
# Venn semasi bize aradaki bagi baglantiyi gosterir.



# venn2

from matplotlib_venn import venn2

pH = data.iloc[:,0]

citric_acid = data.iloc[:,1]

residual_sugar = data.iloc[:,2]

chlorides = data.iloc[:,3]

density = data.iloc[:,4]

# First way to call the 2 group Venn diagram

venn2(subsets = (len(pH)-15, len(citric_acid)-15, 15), set_labels = ('pH', 'citric_acid'))

plt.show()
# donut plot

feature_names = "pH","citric_acid","residual_sugar","density"

feature_size = [len(pH),len(citric_acid),len(residual_sugar),len(density)]

# create a circle for the center of plot

circle = plt.Circle((0,0),0.22,color = "white")

plt.pie(feature_size, labels = feature_names, colors = ["red","green","blue","cyan"] )

p = plt.gcf()

p.gca().add_artist(circle)

plt.title("Rate of Each Materials")

plt.show()
# spider graph

categories = list(data)[1:]

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



values = data.loc[0].drop("quality").values.flatten().tolist()

values = values + values[:1]

ax.plot(angles,values,linewidth = 1,linestyle="solid",label ="pH" )

ax.fill(angles,values,"b",alpha=0.1)



values = data.loc[1].drop("quality").values.flatten().tolist()

values = values + values[:1]

ax.plot(angles,values,linewidth = 1,linestyle="solid",label ="density" )

ax.fill(angles,values,"orange",alpha=0.1)

plt.legend(loc = "upper left",bbox_to_anchor = (0.1,0.1))

plt.show()
# Hangi ozellikler birbiriyle baglantili oldugunu gosteriyor.





# cluster map (dendogram and tree)



df = data.loc[:,["pH","citric_acid","residual_sugar","density"]]

df1 = data.quality

x = dict(zip(df1.unique(),"rgb"))

row_colors = df1.map(x)

cg = sns.clustermap(df,row_colors=row_colors,figsize=(12, 12),metric="correlation")

plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(),rotation = 0,size =8)

plt.show()
# trace1 is line plot

# go: graph object

trace1 = go.Scatter(

    x=df.index,

    y=df.pH,

    mode = "markers",

    xaxis='x2',

    yaxis='y2',

    name = "pH",

    marker = dict(color = 'rgba(0, 112, 20, 0.8)'),

)



# trace2 is histogram

trace2 = go.Histogram(

    x=df.pH,

    opacity=0.75,

    name = "pH",

    marker=dict(color='rgba(10, 200, 250, 0.6)'))



# add trace1 and trace2

data1 = [trace1, trace2]

layout = go.Layout(

    xaxis2=dict(

        domain=[0.7, 1],

        anchor='y2',        

    ),

    yaxis2=dict(

        domain=[0.6, 0.95],

        anchor='x2',

    ),

    title = ' pH Histogram and Scatter Plot'

)

fig = go.Figure(data=data1, layout=layout)

iplot(fig)
# import data again

data = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

# data of iris setosa



data.columns


quality7 = data[data.quality == 7]

# # data of iris virginica

quality8 = data[data.quality == 8]



# trace1 =  iris setosa

trace1 = go.Scatter3d(

    x=quality7.pH,

    y=quality7.density,

    z=quality7.sulphates,

    mode='markers',

    name = "quality-7",

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

    x=quality8.pH,

    y=quality8.density,

    z=quality8.sulphates,

    mode='markers',

    name = "quality-8",

    marker=dict(

        color='rgb(54, 170, 127)',

        size=12,

        line=dict(

            color='rgb(204, 204, 204)',

            width=0.1

        )

    )

)

data2 = [trace1, trace2]

layout = go.Layout(

    title = ' 3D quality-7 and quality-8',

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0

    )

)

fig = go.Figure(data=data2, layout=layout)

iplot(fig)