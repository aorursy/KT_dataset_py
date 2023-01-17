# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from pandas.tools.plotting import parallel_coordinates

# seaborn library
import seaborn as sns

# plotly library
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/column_2C_weka.csv')
data.head()
data.info()
data['class'].value_counts()
x_data = data.drop(['class'], axis=1)
x_data.head()
from sklearn import preprocessing
x = preprocessing.normalize(x_data)
x
y = data['class'].values
# concatenate normalized x_data with class feature of data 
data_new = pd.concat([pd.DataFrame(x), data['class']], axis=1)
data_new.head()
sns.countplot(x="class", data=data)
data.columns
data_new.columns = data.columns
data_new.head()
# Make the plot
plt.figure(figsize=(15,10))
parallel_coordinates(data_new, 'class', colormap=plt.get_cmap("Set1"))
plt.title("class visualization according to Abnormal and Normal")
plt.xlabel("Features of data set")
plt.ylabel("value")
plt.savefig('graph.png')
plt.show()
sns.swarmplot(x="class", y='degree_spondylolisthesis', data=data)
plt.show()
data_new.shape
# import graph objects as "go"
import plotly.graph_objs as go

# creating trace1
trace1 =go.Scatter(
                    x = np.arange(0,310),
                    y = data_new[data_new['class']=='Abnormal'].degree_spondylolisthesis,
                    mode = "markers",
                    name = "Abnormal",
                    marker = dict(color = 'rgba(0, 128, 255, 0.8)'),
                    text= data_new['class'])
# creating trace2
trace2 =go.Scatter(
                    x = np.arange(0,310),
                    y = data_new[data_new['class']=='Normal'].degree_spondylolisthesis,
                    mode = "markers",
                    name = "Normal",
                    marker = dict(color = 'rgba(255, 128, 200, 0.8)'),
                    text= data_new['class'])

data = [trace1, trace2]
layout = dict(title = 'degree_spondylolisthesis',
              xaxis= dict(title= 'index',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Values',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
# import graph objects as "go"
import plotly.graph_objs as go

# creating trace1
trace1 =go.Scatter(
                    x = np.arange(0,310),
                    y = data_new[data_new['class']=='Abnormal'].pelvic_radius,
                    mode = "markers",
                    name = "Abnormal",
                    marker = dict(color = 'rgba(0, 255, 255, 0.8)'),
                    text= data_new['class'])
# creating trace2
trace2 =go.Scatter(
                    x = np.arange(0,310),
                    y = data_new[data_new['class']=='Normal'].pelvic_radius,
                    mode = "markers",
                    name = "Normal",
                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                    text= data_new['class'])

data = [trace1, trace2]
layout = dict(title = 'pelvic_radius',
              xaxis= dict(title= 'index',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Values',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
# import graph objects as "go"
import plotly.graph_objs as go

# creating trace1
trace1 =go.Scatter(
                    x = np.arange(0,310),
                    y = data_new[data_new['class']=='Abnormal'].sacral_slope,
                    mode = "markers",
                    name = "Abnormal",
                    marker = dict(color = 'rgba(0, 100, 255, 0.8)'),
                    text= data_new['class'])
# creating trace2
trace2 =go.Scatter(
                    x = np.arange(0,310),
                    y = data_new[data_new['class']=='Normal'].sacral_slope,
                    mode = "markers",
                    name = "Normal",
                    marker = dict(color = 'rgba(100, 128, 2, 0.8)'),
                    text= data_new['class'])

data = [trace1, trace2]
layout = dict(title = 'sacral_slope',
              xaxis= dict(title= 'index',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Values',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
data_new.columns
# import figure factory
import plotly.figure_factory as ff

data_matrix = data_new.loc[:,["pelvic_incidence","pelvic_tilt numeric", "lumbar_lordosis_angle", 'sacral_slope','pelvic_radius','degree_spondylolisthesis']]
data_matrix["index"] = np.arange(1,len(data_matrix)+1)
# scatter matrix
fig = ff.create_scatterplotmatrix(data_matrix, diag='box', index='index',colormap='Portland',
                                  colormap_type='cat',
                                  height=700, width=700)
iplot(fig)
data_new.head()
data_new['class'] = [1 if i=='Abnormal' else 0 for i in data_new['class']]
y = data_new['class'].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=1)
# KNN
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 3) # randomly selected
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
prediction
# check prediction results with actual correct values, that is, y_test
y_test
print('{} NN score: {}'.format(3,knn.score(x_test,y_test)))
y_test.shape # total number of zeros and ones
62*knn.score(x_test,y_test) # number of correct results out of 114
# find the optimum k value (range is from 1 to 30)
score_list = []
for i in range(1,30):
    knn2 = KNeighborsClassifier(n_neighbors = i) # n_neighbors is our k value
    knn2.fit(x_train, y_train)
    score_list.append(knn2.score(x_test,y_test))

plt.figure(figsize=(10,7))
plt.plot(range(1,30),score_list)
plt.xlabel('k values')
plt.ylabel('accuracy')
plt.show()
# find the optimum k value (range is from 1 to 62)
score_list = []
for i in range(1,62):
    knn2 = KNeighborsClassifier(n_neighbors = i) # n_neighbors is our k value
    knn2.fit(x_train, y_train)
    score_list.append(knn2.score(x_test,y_test))

plt.figure(figsize=(10,7))
plt.plot(range(1,62),score_list)
plt.xlabel('k values')
plt.ylabel('accuracy')
plt.show()