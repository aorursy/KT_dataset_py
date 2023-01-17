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
diabets = pd.read_csv('../input/diabetes.csv')
diabets.info()
 # library
import missingno as msno
dictionary = {"column1" : [np.nan,2,5,4,5,78,7,8,11,10,11,12,29,14,15,26,17,18,145,20,],
            "column2": [7,6,3,56,np.nan,np.nan,7,4,np.nan,10,np.nan,12,13,14,np.nan,16,np.nan,18,np.nan,20],
            "column3" : [8,45,3,58,np.nan,6,89,8,25,10,45,12,13,np.nan,6,8,17,18,np.nan,20]}
d_m = pd.DataFrame(dictionary)
msno.matrix(d_m)
plt.show()
# white space showed missing value, black space showed not missing value.

#we can visialization missing value with bar plot.
msno.bar(d_m, color='maroon')
plt.show()
diabets.head()
data = diabets.drop(['Insulin','DiabetesPedigreeFunction','Pregnancies','BMI'],axis=1)
plt.figure(figsize=(15,10))
parallel_coordinates(data,'Outcome',colormap=plt.get_cmap("Set3"))
plt.title('Breast Cancer')
plt.xlabel("Dataset Feature")
plt.ylabel("cm")
plt.show()
# each color show different feature
corr=diabets.corr()
corr
import networkx as nx
links = corr.stack().reset_index()
#look corelation
links.columns = ['var1', 'var2','value']
#change column names
threshold = 0
#determine threshold value
#we will created filter
links_filtered=links.loc[(links['value'] >=threshold) & (links['var1'] != links['var2'])]

# visualization
G=nx.from_pandas_edgelist(links_filtered, 'var1', 'var2')
nx.draw_circular(G,with_labels=True,node_color = 'purple', node_size = 300, edge_color='pink',linewidths=1, font_size=10)
diabets.head()
# Matplotlib Library (Venn)
from matplotlib_venn import venn2
glucose = data.iloc[:,1]
bloodpressure = data.iloc[:,2]
venn2(subsets = (len(glucose)-50,len(bloodpressure)-50,50), set_labels=('glucose','blood_pressure'))
plt.show()

feature_names='glucose','blood pressure'
feature_size = [len(glucose),len(bloodpressure)]
# determine circle for donut.
# coordinate: 0.0, radius = 0.5
circle = plt.Circle((0,0),0.5,color="white")
plt.pie(feature_size,labels = feature_names,colors=["brown","green"])
# şimdi oluşturduğum circle ı eklemem gerekiyor.
p = plt.gcf()
p.gca().add_artist(circle)
plt.title("glucose and blood pressure")
plt.show()
# Seaborn Library (Cluster )
x = dict(zip(('a','b','c','d','e'),(1,2,3,4,5)))
x
df = data.loc[:,["Glucose","BloodPressure","SkinThickness","Age"]]
df1 = data.Outcome
x = dict(zip(df1.unique(),"rgb"))
row_colors = df1.map(x)
cg = sns.clustermap(df,row_colors=row_colors,figsize=(12, 12),metric="correlation")
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(),rotation=0,size=8)
plt.show()
# Plotly(Inset Plot)
trace1 = go.Scatter(
    x = df.Age,
    y = df.BloodPressure,
    name = "blood pressure",
    xaxis ='x2',
    yaxis ='y2',
    mode = "markers",
    marker = dict(color = 'rgba(0, 11, 20, 0.8)'),
)
trace2 = go.Histogram(
    x = df.Glucose,
    name = "Glucose",
    marker =dict(color='rgba(10, 20, 250, 0.6)'),
    opacity = 0.75,
)
# sonra ikisini birleştirip datamı oluşturuyorum.
data = [trace1,trace2]
layout = go.Layout(
    xaxis2 = dict(domain = [0.7,1], anchor = 'y2'),
    yaxis2 = dict(domain = [0.6,0.95], anchor = 'x2'),
    title = 'Glucose and Blood pressure'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)

# Plotly Library (3D Scatter)
data = pd.read_csv('../input/diabetes.csv')
diabets_zero = data[data.Outcome == 0]
diabets_one = data[data.Outcome == 1]
trace1 = go.Scatter3d(
    x=diabets_zero.Age,
    y=diabets_zero.Glucose,
    z=diabets_zero.BloodPressure,
    mode='markers',
    name = "outcome zero",
    marker=dict(
        color='rgb(217, 100, 100)',
        size=12,
        line=dict(
            color='rgb(255, 255, 255)',
            width=0.1
        )
    )
)
trace2 = go.Scatter3d(
    x=diabets_one.Age,
    y=diabets_one.Glucose,
    z=diabets_one.BloodPressure,
    mode='markers',
    name = "outcome one",
    marker=dict(
        color='rgb(154, 17, 127)',
        size=12,
        line=dict(
            color='rgb(24, 204, 204)',
            width=0.1
        )
    )
)
data = [trace1, trace2]
layout = go.Layout(
    title = ' 3D scatter',
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
