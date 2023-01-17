# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
%matplotlib inline

# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# word cloud library
from wordcloud import WordCloud

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/data.csv")
data.head()
data.info()
data.corr()
#correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data.head()
data.area_mean.plot(kind="line",color="g",label = 'area_mean',linewidth=1,alpha = 0.5,grid = True,linestyle = ':',figsize=(10,10))
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.legend(loc="upper right")
plt.title('Line Plot')            # title = title of plot
plt.show()
data.texture_mean.plot(kind="line",color="r",label = 'texture_mean',linewidth=1,alpha = 0.5,grid = True,linestyle = '-.',figsize=(10,10))
plt.legend(loc="upper right")
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
plt.subplots(figsize=(10,10))
plt.plot(data.texture_mean[0:100],linestyle="-.")
plt.plot(data.radius_mean[0:100],linestyle="-")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("Texture Mean-Radius Mean Plot for first 100 items")
plt.legend(loc="upper right")
plt.show()
# smoothness_mean-compactness_mean plot
plt.subplots(figsize=(10,10))
plt.plot(data.smoothness_mean[0:100],linestyle="-.")
plt.plot(data.compactness_mean[0:100],linestyle="-")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.grid()
plt.title("smoothness_mean-compactness_mean plot")
plt.legend(loc="upper right")
plt.show()
radius=data.radius_mean
texture=data.texture_mean
compactness=data.compactness_mean
smoothness=data.smoothness_mean
plt.subplots(figsize=(10,10))

plt.subplot(4,1,1)
plt.title("radius_mean-texture_mean-compactness_mean subplot")
plt.plot(radius,color="r",label="radius_mean")
plt.legend()
plt.grid()

plt.subplot(4,1,2)
plt.plot(texture,color="b",label="texture_mean")
plt.legend()
plt.grid()

plt.subplot(4,1,3)
plt.plot(compactness,color="g",label="compactness_mean")
plt.legend()
plt.grid()

plt.subplot(4,1,4)
plt.plot(smoothness,color="purple",label="smoothness_mean")
plt.legend()
plt.grid()

plt.show()



#area mean vs compactness_mean for first 10 items
plt.subplots(figsize=(8,8))
plt.plot(data["area_mean"][0:10],data["compactness_mean"][0:10],color="lime",alpha = 0.5, linestyle = ':')
plt.xlabel("area_mean")
plt.ylabel("compactness_mean")
plt.title("compactness_mean vs area_mean report")
plt.show()

#histogram of radius_mean
data.radius_mean.plot(kind="hist",bins=10,figsize=(10,10),color="b",grid="True")
plt.xlabel("radius_mean")
plt.legend(loc="upper right")
plt.title("Radius_mean Histogram")
plt.show()
#histogram of area_mean
data.area_mean.plot(kind="hist",bins=10,figsize=(10,10),color="r",grid="True")
plt.xlabel("area_mean")
plt.legend(loc="upper right")
plt.title("Area_mean Histogram")
plt.show()
#histogram of texture_mean
data.texture_mean.plot(kind="hist",bins=10,figsize=(10,10),color="lime",grid="True")
plt.xlabel("texture_mean")
plt.legend(loc="upper right")
plt.title("texture_mean Histogram")
plt.show()

# histogram subplot with non cumulative and cumulative
fig, axes = plt.subplots(nrows=2,ncols=1)
data.plot(kind="hist",y="texture_mean",bins = 50,range= (0,50),normed = True,ax = axes[0])
data.plot(kind = "hist",y = "texture_mean",bins = 50,range= (0,50),normed = True,ax = axes[1],cumulative = True)
plt.savefig('graph.png')
plt.show()
data.diagnosis.unique()
# Lets look frequency of diagnosis types
print(data.diagnosis.value_counts(dropna=False))
# diagnosis-radius_mean bar plot
plt.subplots(figsize=(10,10))
plt.bar(data.diagnosis,data.radius_mean,color="r")
plt.xlabel("diagnosis")
plt.ylabel("radius_mean")
plt.title("diagnosis-radius_mean bar plot")
plt.show()
# diagnosis-area_mean bar plot
plt.subplots(figsize=(10,10))
plt.bar(data.diagnosis,data.area_mean,color="b")
plt.xlabel("diagnosis")
plt.ylabel("area_mean")
plt.title("diagnosis-area_mean bar plot")
plt.show()
# diagnosis-smoothness_mean bar plot
plt.subplots(figsize=(10,10))
plt.bar(data.diagnosis,data.smoothness_mean,color="r")
plt.xlabel("diagnosis")
plt.ylabel("smoothness_mean")
plt.title("diagnosis-smoothness_mean bar plot")
plt.show()
# mean of texture_mean given diagnosis
diagnosis_list=list(data.diagnosis.unique())
diagnosis_texture=[]
for i in diagnosis_list:
    x=data[data.diagnosis==i]
    d_texture=sum(x.texture_mean)/len(x)
    diagnosis_texture.append(d_texture)
    
data1=pd.DataFrame({"diagnosis":diagnosis_list,"texture":diagnosis_texture})
new_index=(data1["texture"].sort_values(ascending=False)).index.values
sorted_data=data1.reindex(new_index)

# visualization
plt.figure(figsize=(15,10)) #create a new figure with size (15,10). 
sns.barplot(x=sorted_data['diagnosis'], y=sorted_data['texture'])
plt.xticks(rotation= 360)
plt.xlabel('Diagnosis')
plt.ylabel('Texture Rate')
plt.title('Texture Rate Given Diagnosis')
plt.show()
# Percentage of Radius Mean and Texture Mean According to Diagnosis
#horizontal bar plot

f,ax = plt.subplots(figsize = (9,9)) #create a figure of 9x9.
sns.barplot(x=data.radius_mean,y=data.diagnosis,color='green',alpha = 0.5,label='radius_mean' )
sns.barplot(x=data.texture_mean,y=data.diagnosis,color='lime',alpha = 0.7,label='texture_mean')

ax.legend(loc='lower right',frameon = True)
ax.set(xlabel='Percentage of Diagnosis', ylabel='Diagnosis',title = "Percentage of Radius Mean and Texture Mean According to Diagnosis ")
# texture_mean area_mean Scatter Plot
plt.subplots(figsize=(8,8))
plt.scatter(data.area_mean,data.texture_mean,color="g",alpha=0.5)
plt.xlabel("area_mean")
plt.ylabel("texture_mean")
plt.title('texture_mean area_mean Scatter Plot')
plt.grid()
# smoothness_mean compactness_mean Scatter Plot
plt.subplots(figsize=(8,8))
plt.scatter(data.smoothness_mean,data.compactness_mean,color="r",alpha=0.5)
plt.xlabel("smoothness_mean")
plt.ylabel("compactness_mean")
plt.title('smoothness_mean compactness_mean Scatter Plot')
plt.grid()
# or we can find the above graph by 
data.plot(kind="scatter",x="smoothness_mean",y="compactness_mean",figsize=(8,8),grid=True)
plt.show()
#compare area mean by diagnosis
data.boxplot(column="area_mean",by="diagnosis")
plt.show()
#compare texture_mean by diagnosis
data.boxplot(column="texture_mean",by="diagnosis")
plt.show()
# texture mean and compatness mean of each diagnosis
# normalizing data
data2=data.copy()
data2["texture_mean"]=data2["texture_mean"]/max(data2["texture_mean"])
data2["compactness_mean"]=data2["compactness_mean"]/max(data2["compactness_mean"])

# visualize
f,ax1 = plt.subplots(figsize =(10,10))
sns.pointplot(x=data2.diagnosis,y=data2["texture_mean"],color='lime',alpha=0.8)
sns.pointplot(x=data2.diagnosis,y=data2["compactness_mean"],color='red',alpha=0.8)
plt.text(0.8,0.4,'compactness mean',color='red',fontsize = 17,style = 'italic')
plt.text(0.8,0.42,'texture mean',color='lime',fontsize = 18,style = 'italic')
plt.xlabel('Diagnosis',fontsize = 15,color='blue')
plt.ylabel('Values',fontsize = 15,color='blue')
plt.title('texture_mean vs  compactness_mean',fontsize = 20,color='blue')
plt.grid()
# texture_mean vs area_mean in terms of id

# normalizing data
data2=data.loc[0:9,["id","texture_mean","area_mean","smoothness_mean"]]
data2["texture_mean"]=data2["texture_mean"]/max(data2["texture_mean"])
data2["area_mean"]=data2["area_mean"]/max(data2["area_mean"])

# visualize
f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x='id',y='texture_mean',data=data2,color='lime',alpha=0.8)
sns.pointplot(x='id',y='area_mean',data=data2,color='red',alpha=0.8)
plt.text(0.4,0.45,'texture_mean',color='lime',fontsize = 18,style = 'italic')
plt.text(0.4,0.4,'area_mean',color='red',fontsize = 17,style = 'italic')

plt.xlabel('id',fontsize = 15,color='blue')
plt.ylabel('Values',fontsize = 15,color='blue')
plt.title('texture_mean  vs  area_mean',fontsize = 20,color='blue')
plt.grid()
plt.show()
data.head()
# texture_mean vs area_mean in terms of id

# import graph objects as "go"
import plotly.graph_objs as go

#preapering data
data3=data.loc[0:9,["id","texture_mean","compactness_mean","smoothness_mean"]]


# Creating trace1
trace1 = go.Scatter(
                    x = data3.id,
                    y = data3["smoothness_mean"],
                    mode = "lines",
                    name = "smoothness_mean",
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
                    text= data3.id)
# Creating trace2
trace2 = go.Scatter(
                    x = data3.id,
                    y = data3["compactness_mean"],
                    mode = "lines+markers",
                    name = "compactness_mean",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    text= data3.id)
dat = [trace1, trace2]
layout = dict(title = 'smoothness_mean and compactness_mean vs id',
              xaxis= dict(title= 'id',ticklen= 5,zeroline= False)
             )
fig = dict(data = dat, layout = layout)
iplot(fig)
#texture mean and area mean
g = sns.jointplot(data.texture_mean, data.area_mean, kind="kde", size=7)
plt.savefig('graph.png')
plt.show()
# you can change parameters of joint plot
# kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }
# Different usage of parameters but same plot with previous one 
g = sns.jointplot("area_mean", "texture_mean", data=data,size=7, ratio=3, color="r")
sns.kdeplot(data.area_mean, data.texture_mean, shade=True, cut=3) #shade=True: grafikteki sekillerin ici dolu olsun. cut=3: cıkan sekillerin buyuklugunu ayarlar.
plt.show()
data3=data.loc[0:100,["texture_mean","area_mean"]]
sns.pairplot(data3)
plt.show()

sns.countplot(data.diagnosis)
plt.title("diagnosis",color = 'red',fontsize=15)
plt.show()
above20=["big" if i>20 else "small" for i in data.texture_mean ]
df=pd.DataFrame({"texture":above20})
sns.countplot(df.texture)
plt.title("size of texture")
plt.show()
sns.countplot(data.radius_mean[0:10])
plt.title("number of radius_mean of first 10 items",color="b",fontsize=12)
plt.show()
# prepare data frames
data4=data.iloc[:2,:]

# import graph objects as "go"
import plotly.graph_objs as go

# create trace1 
trace1 = go.Bar(
                x = data4.id,
                y = data4["radius_mean"],
                name = "radius_mean",
                marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = data4.id)
# create trace2 
trace2 = go.Bar(
                x = data4.id,
                y =  data4["texture_mean"],
                name = "texture_mean",
                marker = dict(color = 'rgba(255, 255, 128, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = data4.id)
dat = [trace1, trace2]
layout = go.Layout(barmode = "group")
fig = go.Figure(data = dat, layout = layout)
iplot(fig)
# import graph objects as "go"
import plotly.graph_objs as go

x = data4.id

trace1 = {
  'x': x,
  'y': data4["radius_mean"],
  'name': 'radius_mean',
  'type': 'bar'
};
trace2 = {
  'x': x,
  'y': data4["texture_mean"],
  'name': 'texture_mean',
  'type': 'bar'
};
dat = [trace1, trace2];
layout = {
  'xaxis': {'title': 'First Two Sample'},
  'barmode': 'relative',
  'title': 'radius_mean and texture_mean of first two sample'
};
fig = go.Figure(data = dat, layout = layout)
iplot(fig)
data.head()
labels=data.diagnosis.value_counts()
colors=["grey","blue"]
explode=[0,0]
sizes=data.diagnosis.value_counts().values

plt.figure(figsize = (7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title("Diagnosis in Data",color = 'blue',fontsize = 15)
plt.show()
pie1_list=data.diagnosis.value_counts().values
labels = data.diagnosis.value_counts().index
# figure
fig = {
  "data": [
    {
      "values": pie1_list,
      "labels": labels,
      "domain": {"x": [0, .5]},
      "name": "diagnosis",
      "hoverinfo":"label+percent",
      "hole": .3,
      "type": "pie"
    },],
  "layout": {
        "title":"Diagnosis Type",
        "annotations": [
            { "font": { "size": 20},
              "showarrow": False,
              "text": "diagnosis",
                "x": 0.20,
                "y": 1
            },
        ]
    }
}
iplot(fig)
df=data.copy()
df.drop(["id","Unnamed: 32"],axis=1,inplace=True)

df.diagnosis=[1 if each=="M" else 0 for each in df.diagnosis]

y=df.diagnosis.values
x_data=df.drop(["diagnosis"],axis=1)
#normalization
# (x-min(x))/(max(x)-min(x))

x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data)).values

#train test split

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42) # randomlık id si=42, yani tutarlı bir sekilde bol demek. 

#featurlarım row olmalı.
x_train=x_train.T
x_test=x_test.T
y_train=y_train.T
y_test=y_test.T

print("x_train: ", x_train.shape)
print("x_test: ", x_test.shape)
print("y_train: ", y_train.shape)
print("y_test: ", y_test.shape)
#parameter initialize and sigmoid function
# dimension=30

def initialize_weights_and_bias(dimension):
    
    w=np.full((dimension,1),0.01) # (dimension)x1 lik 0.01 lerden oluşan bir matris.
    b=0.0
    return w,b

def sigmoid(z):
    
    y_head=1/(1+np.exp(-z))
    return y_head
# forward_backward_propagation
def forward_backward_propagation(w,b,x_train,y_train):
    #forward propagation
    z=np.dot(w.T,x_train)+b
    y_head=sigmoid(z)
    loss=-y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost=(np.sum(loss))/x_train.shape[1] # x_train.shape[1] is for scaling
    
    #backward propagation
    derivative_weight=(np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1] is for scaling
    derivative_bias=np.sum(y_head-y_train)/x_train.shape[1]                   # x_train.shape[1] is for scaling
    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}
    
    return cost,gradients

# Updating(learning) parameters
def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):
    cost_list = []
    cost_list2 = []
    index = []
    # updating(learning) parameters is number_of_iterarion times
    for i in range(number_of_iterarion):
        # make forward and backward propagation and find cost and gradients
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        # lets update
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
    # we update(learn) parameters weights and bias
    parameters = {"weight": w,"bias": b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iteration")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list
#prediction

def predict(w,b,x_test):
    # x_test is an input for forward propagation
    z = sigmoid(np.dot(w.T,x_test)+b)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction

#logistic regression
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):
    # initialize
    dimension =  x_train.shape[0]  # that is 30
    w,b = initialize_weights_and_bias(dimension)
    # do not change learning rate
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)
    # y_prediction_train = predict(parameters["weight"],parameters["bias"],x_train)

    # Print train/test Errors
    #print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 30)
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 100)
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 2, num_iterations = 300)
# sklearn with LR

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()

lr.fit(x_train.T,y_train.T)
print("test accuracy {}".format(lr.score(x_test.T,y_test.T)))
