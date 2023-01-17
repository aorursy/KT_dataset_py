# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#We read data

data = pd.read_csv("../input/column_2C_weka.csv")
#We ignore this

a = data.degree_spondylolisthesis.max()

data[data.degree_spondylolisthesis == a] = np.mean(data.degree_spondylolisthesis)
#The feature of our data(if you dont know feature it is like header)

data.columns
import matplotlib.pyplot as plt 

#We use sacral_radius and pelvic incidence for showing our cluster.

plt.scatter(x = data["pelvic_radius"],y = data["degree_spondylolisthesis"],color = "black")

x = data["pelvic_radius"]

y = data["degree_spondylolisthesis"]

#we use two columns for clustering our data.The computer see our data so 
#we can draw with class.It is original class

plt.scatter(x = data[data["class"] == "Normal"].pelvic_radius , y = data[data["class"] == "Normal"].degree_spondylolisthesis,color = "red")

plt.scatter(x = data[data["class"] == "Abnormal"].pelvic_radius , y = data[data["class"] == "Abnormal"].degree_spondylolisthesis,color = "yellow")

#probably we use two columns for clustering
#We implement our model so. We need just one data

cluster_data_arg = {"x":x,"y":y}

cluster_data = pd.DataFrame(cluster_data_arg)

from sklearn.cluster import KMeans

wcss = []

for k in range(1,10):

    kmeans = KMeans(n_clusters = k)

    kmeans.fit(cluster_data)

    wcss.append(kmeans.inertia_)

plt.plot(range(1,10),wcss,color="blue")

plt.show()

#Number of cluster should be 2.We look the most increase where(you can look "elbow" rules)
#if we want to add label to data We do cluster data(x and y)

kmeans2 = KMeans(n_clusters = 2)

clusters = kmeans2.fit_predict(cluster_data)

data["label"] = clusters
#We look how many whether true our clusters

data_ac = pd.read_csv("../input/column_2C_weka.csv")

data_ac["class"] = [0 if each == "Abnormal" else 1 for each in data_ac["class"]]

data_ac_class = data_ac["class"]

predict_class = data["label"]

print("accuracy is : {}".format(100 - np.mean(np.abs(data_ac_class - predict_class)*100)))

kmeans_accuracy = int(100 - np.mean(np.abs(data_ac_class - predict_class)*100))
#We draw our data with clusters.We have 0,1,2 label

plt.scatter(x = data[data.label == 0].pelvic_radius,y = data[data.label == 0].degree_spondylolisthesis , color = "yellow")

plt.scatter(x = data[data.label == 1].pelvic_radius,y = data[data.label == 1].degree_spondylolisthesis , color = "blue")
from scipy.cluster.hierarchy import linkage, dendrogram

merg = linkage(cluster_data,method="ward")

dendrogram(merg,leaf_rotation = 90)

plt.xlabel("data points")

plt.ylabel("euclidean distance")

plt.show()
from sklearn.cluster import AgglomerativeClustering



hiyerartical_cluster = AgglomerativeClustering(n_clusters = 2,affinity= "euclidean",linkage = "ward")

cluster = hiyerartical_cluster.fit_predict(cluster_data)



data["label_hc"] = cluster
plt.scatter(x = data[data.label_hc == 0].pelvic_radius,y = data[data.label_hc == 0].degree_spondylolisthesis , color = "yellow")

plt.scatter(x = data[data.label_hc == 1].pelvic_radius,y = data[data.label_hc == 1].degree_spondylolisthesis , color = "blue")
#Let's we calculate accuracy.

#data_ac["class"] = [0 if each == "Abnormal" else 1 for each in data_ac["class"]]

#data_ac_class = data_ac["class"]

#predict_class = data["label"]

print("accuracy is : {}".format(100 - np.mean(np.abs(data_ac_class - data["label_hc"])*100)))

hc_accuracy = int(100 - np.mean(np.abs(data_ac_class - data["label_hc"])*100))
#We compare our model in graph

fig = plt.figure(figsize = (15,5))



plt.subplot(1, 3, 1)

plt.scatter(x = data[data["class"] == "Normal"].pelvic_radius , y = data[data["class"] == "Normal"].degree_spondylolisthesis,color = "red")

plt.scatter(x = data[data["class"] == "Abnormal"].pelvic_radius , y = data[data["class"] == "Abnormal"].degree_spondylolisthesis,color = "yellow")

plt.title("original class")



plt.subplot(1, 3, 2)

plt.scatter(x = data[data.label == 0].pelvic_radius,y = data[data.label == 0].degree_spondylolisthesis , color = "yellow")

plt.scatter(x = data[data.label == 1].pelvic_radius,y = data[data.label == 1].degree_spondylolisthesis , color = "red")

plt.title("kmeans") 



plt.subplot(1, 3, 3)

plt.scatter(x = data[data.label_hc == 0].pelvic_radius,y = data[data.label_hc == 0].degree_spondylolisthesis , color = "yellow")

plt.scatter(x = data[data.label_hc == 1].pelvic_radius,y = data[data.label_hc == 1].degree_spondylolisthesis , color = "red")

plt.title("hierarchical")



plt.show()
#We have hc_accuracy and kmeans_accuracy

list1 = ["hc","kmeans"]

list2 = [hc_accuracy,kmeans_accuracy]

list3 = [100 - hc_accuracy,100 - kmeans_accuracy]

dictionary = {"name":list1,"value":list2,"hundred":list3}

dt = pd.DataFrame(dictionary)
import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

trace1 = go.Bar(

x = dt.name,

y = dt.value,

name="accuracy",

marker = {"color":"rgba(111,23,155,0.5)"},

text=dt.name

)

trace2 = go.Bar(

x = dt.name,

y = dt.hundred,

name="mistake",

marker = {"color":"rgba(47,69,187)"},

)

fig_data = [trace1,trace2]

layout = go.Layout(barmode = "relative")

fig = go.Figure(data = fig_data, layout = layout)

iplot(fig)
