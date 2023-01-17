#load the necessary libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



palette = sns.color_palette("Paired")



import warnings

warnings.filterwarnings("ignore")
data = pd.read_csv('../input/Mall_Customers.csv')

data.head()
data = data.rename(index=str,columns = {"CustomerID":"id","Annual Income (k$)":"income","Spending Score (1-100)":"score"})
print(data.columns)

print(type(data))
print(data.describe())

print(data.isnull().sum())
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



data['Gender'] = le.fit_transform(data.Gender)

print(le.classes_)

#Male is encoded as 1

#Female is encoded as 0.

print(data.head(5))
m,f=data['Gender'].value_counts()[1],data['Gender'].value_counts()[0]

gen = [m,f]

lab = ['Male','Female']

plt.pie(gen,labels=lab, shadow=True,autopct='%1.0f%%', startangle=140)



fig = plt.figure()

fig.suptitle('PDF',fontsize=20)

fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i in range(2,len(data.columns)):

    ax = fig.add_subplot(1, 3, i-1)

    fig.set_figheight(5)

    fig.set_figwidth(20)

    sns.distplot(data[str(data.columns[i])],hist=False,kde_kws={"shade": True},color='orange')

    

    ax.title.set_text(data.columns[i])

    



plt.show()

fig = plt.figure()

fig.suptitle('CDF',fontsize=20)

fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i in range(2,len(data.columns)):

    ax = fig.add_subplot(1, 3, i-1)

    fig.set_figheight(5)

    fig.set_figwidth(20)

    sns.distplot(data[str(data.columns[i])],hist=False,kde_kws={"shade": True,"cumulative":True},color='g')

    ax.title.set_text(data.columns[i])

    



plt.show()
fig = plt.figure()



fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i in range(2,len(data.columns)-1):

    sns.jointplot(str(data.columns[i]),'score',data, kind="kde",space=0, color="#4CB391")
fig = plt.figure()

fig.suptitle('Violin Plots',fontsize=20)

fig.subplots_adjust(hspace=0.4, wspace=0.4)

cols = ['Age', 'income', 'score']

for i in range(len(cols)):

    ax = fig.add_subplot(1, 3, i+1)

    fig.set_figheight(5)

    fig.set_figwidth(20)

    sns.violinplot(x = 'Gender' , y = cols[i], data = data )

    

    
fig = plt.figure()

fig.suptitle('Violin Plots',fontsize=20)

fig.subplots_adjust(hspace=0.4, wspace=0.4)

cols = ['Age', 'income', 'score']

for i in range(len(cols)):

    ax = fig.add_subplot(1, 3, i+1)

    fig.set_figheight(5)

    fig.set_figwidth(20)

    sns.violinplot(x = 'Gender' , y = cols[i], data = data )

    
import itertools

col_list = ['Age','score','income','Gender']

combinations =list(itertools.permutations(col_list, 3))

print(combinations)

fig = plt.figure()

fig.suptitle('Scatter Plots',fontsize=20)

fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i in range(0,len(combinations)):

    try:

        ax = fig.add_subplot(4, 4, i)

        fig.set_figheight(20)

        fig.set_figwidth(20)

        sns.scatterplot(x=str(combinations[i][0]),y=str(combinations[i][1]),hue=str(combinations[i][2]),data=data)

    except:

        pass

print(data.columns)

attr,label = data.iloc[:,1:-1],data.iloc[:,-1]

sse = {}

from sklearn.cluster import KMeans

for k in range(1, 10):

    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(attr)

    

    sse[k] = kmeans.inertia_ 

plt.plot(list(sse.keys()), list(sse.values()))

plt.xlabel("Number of cluster")

plt.ylabel("SSE")

plt.show()

kmeans = KMeans(n_clusters=4).fit(attr)

label = kmeans.labels_

centers = kmeans.cluster_centers_
import plotly as py

import plotly.graph_objs as go

import plotly.offline as py_of

py_of.init_notebook_mode(connected=True)

data['labels3'] =  label

trace1 = go.Scatter3d(

    x= data['Age'],

    y= data['score'],

    z= data['income'],

    mode='markers',

     marker=dict(

        color = data['labels3'], 

        size= 20,

        line=dict(

            color= data['labels3'],

            width= 12

        ),

    

        

     )

)

data_2 = [trace1]

layout = go.Layout(

    

    title= 'Clusters',

    scene = dict(

            xaxis = dict(title  = 'Age'),

            yaxis = dict(title  = 'Spending Score'),

            zaxis = dict(title  = 'Annual Income')

        )

)

layout.autosize= True

fig = go.Figure(data=data_2, layout=layout)

py_of.iplot(fig)
from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y = train_test_split(attr,label)

data_temp = data.iloc[:,[2,4]]

sse = {}

from sklearn.cluster import KMeans

for k in range(1, 10):

    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(data_temp)

    #print(data["clusters"])

    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center

plt.figure()

plt.plot(list(sse.keys()), list(sse.values()))

plt.xlabel("Number of cluster")

plt.ylabel("SSE")

plt.show()
kmeans = KMeans(n_clusters=4,random_state=2).fit(data_temp)

label = kmeans.labels_

centers = kmeans.cluster_centers_

print(data_temp.head())

sns.scatterplot(x='score',y='Age',hue = label,data = data_temp,palette="Set2")
data_temp = data.iloc[:,[3,4]]

sse = {}

from sklearn.cluster import KMeans

for k in range(1, 10):

    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(data_temp)

    #print(data["clusters"])

    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center

plt.figure()

plt.plot(list(sse.keys()), list(sse.values()))

plt.xlabel("Number of cluster")

plt.ylabel("SSE")

plt.show()
kmeans = KMeans(n_clusters=5,init='k-means++',max_iter=1000,random_state=2).fit(data_temp)

label = kmeans.labels_

centers = kmeans.cluster_centers_
sns.scatterplot(x='score',y='income',data=data_temp,hue=label)