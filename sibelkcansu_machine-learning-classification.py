# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
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
data=pd.read_csv("../input/xAPI-Edu-Data.csv")
data.head()
data.info()
data.describe()
#heatmap
f,ax=plt.subplots(figsize=(8,8))
sns.heatmap(data.corr(),annot=True,linewidths=0.5,linecolor="red",fmt=".1f",ax=ax)
plt.show()
data.head()
# Line Plot
data.raisedhands.plot(kind="line",color="g",label = 'raisedhands',linewidth=1,alpha = 0.5,grid = True,linestyle = ':',figsize=(10,10))
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.legend(loc="upper right")
plt.title('Line Plot')            # title = title of plot
plt.show()
plt.subplots(figsize=(10,10))
plt.plot(data.raisedhands[:100],linestyle="-.")
plt.plot(data.VisITedResources[:100],linestyle="-")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("Raidehands and VisITedResources Line Plot")
plt.legend(loc="upper right")
plt.show()
#subplots
raisedhands=data.raisedhands
VisITedResources=data.VisITedResources

plt.subplots(figsize=(10,10))
plt.subplot(2,1,1)
plt.title("raisedhands-VisITedResources subplot")
plt.plot(raisedhands[:100],color="r",label="raisedhands")
plt.legend()
plt.grid()

plt.subplot(2,1,2)
plt.plot(VisITedResources[:100],color="b",label="VisITedResources")
plt.legend()
plt.grid()

plt.show()
# discussion and raisedhands line plot in plotly
# import graph objects as "go"
import plotly.graph_objs as go

# Creating trace1
trace1 = go.Scatter(
                    x = np.arange(0,82),
                    y = data.Discussion,
                    mode = "lines",
                    name = "discussion",
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
                    )
# Creating trace2
trace2 = go.Scatter(
                    x =np.arange(0,82) ,
                    y = data.raisedhands,
                    mode = "lines",
                    name = "raisedhands",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                   )
df = [trace1, trace2]
layout = dict(title = 'Discussion and Raisedhands of Students',
              xaxis= dict(title= 'raisedhands',ticklen= 5,zeroline= False)
             )
fig = dict(data = df, layout = layout)
iplot(fig)
#histogram of raisedhands
data.raisedhands.plot(kind="hist",bins=10,figsize=(10,10),color="b",grid="True")
plt.xlabel("raisedhands")
plt.legend(loc="upper right")
plt.title("raisedhands Histogram")
plt.show()
# histogram subplot with non cumulative and cumulative
fig, axes = plt.subplots(nrows=2,ncols=1)
data.plot(kind="hist",y="raisedhands",bins = 50,range= (0,50),normed = True,ax = axes[0])
data.plot(kind = "hist",y = "raisedhands",bins = 50,range= (0,50),normed = True,ax = axes[1],cumulative = True)
plt.savefig('graph.png')
plt.show()
#raidehands vs Discussion scatter plot
plt.subplots(figsize=(10,10))
plt.scatter(data.raisedhands,data.Discussion,color="green")
plt.xlabel("raisedhands")
plt.ylabel("Discussion")
plt.grid()
plt.title("Raidehands vs Discussion Scatter Plot",color="red")
plt.show()
#raidehands vs AnnouncementsView scatter plot
color_list1 = ['red' if i=='M' else 'blue' for i in data.gender]
plt.subplots(figsize=(10,10))
plt.scatter(data.raisedhands,data.AnnouncementsView,color=color_list1, alpha=0.8)
plt.xlabel("raisedhands")
plt.ylabel("AnnouncementsView")
plt.grid()
plt.title("Raidehands vs Announcements View Scatter Plot",color="black",fontsize=15)
plt.show()
len(data.raisedhands.unique())
# raisedhands  in terms of gender

# import graph objects as "go"
import plotly.graph_objs as go

# creating trace1
trace1 =go.Scatter(
                    x = np.arange(0,82),
                    y = data[data.gender=='M'].raisedhands,
                    mode = "markers",
                    name = "male",
                    marker = dict(color = 'rgba(0, 100, 255, 0.8)'),
                    )
# creating trace2
trace2 =go.Scatter(
                    x = np.arange(0,82),
                    y = data[data.gender=="F"].raisedhands,
                    mode = "markers",
                    name = "female",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    )

df = [trace1, trace2]
layout = dict(title = 'raisedhands',
              xaxis= dict(title= 'index',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Values',ticklen= 5,zeroline= False)
             )
fig = dict(data = df, layout = layout)
iplot(fig)
# Discussion  in terms of gender

# import graph objects as "go"
import plotly.graph_objs as go

# creating trace1
trace1 =go.Scatter(
                    x = np.arange(0,82),
                    y = data[data.gender=='M'].Discussion,
                    mode = "markers",
                    name = "male",
                    marker = dict(color = 'rgba(0, 100, 255, 0.8)'),
                    text= data[data.gender=="M"].gender)
# creating trace2
trace2 =go.Scatter(
                    x = np.arange(0,82),
                    y = data[data.gender=="F"].Discussion,
                    mode = "markers",
                    name = "female",
                    marker = dict(color = 'rgba(200, 50, 150, 0.8)'),
                    text= data[data.gender=="F"].gender)

df = [trace1, trace2]
layout = dict(title = 'Discussion',
              xaxis= dict(title= 'index',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Values',ticklen= 5,zeroline= False)
             )
fig = dict(data = df, layout = layout)
iplot(fig)
# Plotting Scatter Matrix
color_list = ['red' if i=='M' else 'green' for i in data.gender]
pd.plotting.scatter_matrix(data.loc[:, data.columns != 'gender'],
                                       c=color_list,
                                       figsize= [15,15],
                                       diagonal='hist',
                                       alpha=0.8,
                                       s = 200,
                                       marker = '.',
                                       edgecolor= "black")

plt.show()
# Raisehands Average in terms of Topic

# we will create a data containing averages of the numerical values of our data.
topic_list=list(data.Topic.unique())
rh_av=[]
d_av=[]
aview_av=[]
vr_av=[]
for i in topic_list:
    rh_av.append(sum(data[data["Topic"]==i].raisedhands)/len(data[data["Topic"]==i].raisedhands))
    d_av.append(sum(data[data["Topic"]==i].Discussion)/len(data[data["Topic"]==i].Discussion))
    aview_av.append(sum(data[data["Topic"]==i].AnnouncementsView)/len(data[data["Topic"]==i].AnnouncementsView))
    vr_av.append(sum(data[data["Topic"]==i].VisITedResources)/len(data[data["Topic"]==i].VisITedResources))
data2=pd.DataFrame({"topic":topic_list,"raisedhands_avg":rh_av,"discussion_avg":d_av,"AnnouncementsView_avg":aview_av, "VisITedResources_avg":vr_av})

# we will sort data2 interms of index of raisedhands_avg in ascending order
new_index2 = (data2['raisedhands_avg'].sort_values(ascending=True)).index.values 
sorted_data2 = data2.reindex(new_index2)
sorted_data2.head()

# visualization
plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data2['topic'], y=sorted_data2['raisedhands_avg'])
plt.xticks(rotation= 90)
plt.xlabel('Topics')
plt.ylabel('Raisehands Average')
plt.title("Raisehands Average in terms of Topic")
# horizontal bar plot
# Raised hands, Discussion and Announcements View averages acording to topics

f,ax = plt.subplots(figsize = (9,15)) #create a figure of 9x15 .
sns.barplot(x=rh_av,y=topic_list,color='cyan',alpha = 0.5,label='Raised hands' )
sns.barplot(x=d_av,y=topic_list,color='blue',alpha = 0.7,label='Discussion')
sns.barplot(x=aview_av,y=topic_list,color='red',alpha = 0.6,label='Announcements View')

ax.legend(loc='upper right',frameon = True)
ax.set(xlabel='Average ', ylabel='Topics',title = "Average of Numerical Values of Data According to Topics ")
# raisehands and discussion average acording to topic

# we will sort data2 interms of index of raisedhands_avg in descending order
new_index3 = (data2['raisedhands_avg'].sort_values(ascending=False)).index.values 
sorted_data3 = data2.reindex(new_index3)

# create trace1 
trace1 = go.Bar(
                x = sorted_data3.topic,
                y = sorted_data3.raisedhands_avg,
                name = "raisedhands average",
                marker = dict(color = 'rgba(255, 174, 155, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = sorted_data3.topic)
# create trace2 
trace2 = go.Bar(
                x = sorted_data3.topic,
                y = sorted_data3.discussion_avg,
                name = "discussion average",
                marker = dict(color = 'rgba(255, 255, 128, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = sorted_data3.topic)
df= [trace1, trace2]
layout = go.Layout(barmode = "group",title= "Discussion and Raisedhands Average of Each Topic")
fig = go.Figure(data = df, layout = layout)
iplot(fig)
# raisehands and discussion average acording to PlaceofBirth

#prepare data
place_list=list(data.PlaceofBirth.unique())
rh_av=[]
d_av=[]
aview_av=[]
vr_av=[]
for i in place_list:
    rh_av.append(sum(data[data["PlaceofBirth"]==i].raisedhands)/len(data[data["PlaceofBirth"]==i].raisedhands))
    d_av.append(sum(data[data["PlaceofBirth"]==i].Discussion)/len(data[data["PlaceofBirth"]==i].Discussion))
    aview_av.append(sum(data[data["PlaceofBirth"]==i].AnnouncementsView)/len(data[data["PlaceofBirth"]==i].AnnouncementsView))
    vr_av.append(sum(data[data["PlaceofBirth"]==i].VisITedResources)/len(data[data["PlaceofBirth"]==i].VisITedResources))
data4=pd.DataFrame({"PlaceofBirth":place_list,"raisedhands_avg":rh_av,"discussion_avg":d_av,"AnnouncementsView_avg":aview_av, "VisITedResources_avg":vr_av})

new_index4=data4["raisedhands_avg"].sort_values(ascending=False).index.values
sorted_data4=data4.reindex(new_index4)

# create trace1 
trace1 = go.Bar(
                x = sorted_data4.PlaceofBirth,
                y = sorted_data4.raisedhands_avg,
                name = "raisedhands average",
                marker = dict(color = 'rgba(200, 125, 200, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = sorted_data4.PlaceofBirth)
# create trace2 
trace2 = go.Bar(
                x = sorted_data4.PlaceofBirth,
                y = sorted_data4.discussion_avg,
                name = "discussion average",
                marker = dict(color = 'rgba(128, 255, 128, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = sorted_data4.PlaceofBirth)
df= [trace1, trace2]
layout = go.Layout(barmode = "group",title= "Discussion and Raisedhands Average acording to PlaceofBirth")
fig = go.Figure(data = df, layout = layout)
iplot(fig)
trace1 = {
  'x': sorted_data4.PlaceofBirth,
  'y': sorted_data4.raisedhands_avg,
  'name': 'raisedhands average',
  'type': 'bar'
};
trace2 = {
  'x': sorted_data4.PlaceofBirth,
  'y': sorted_data4.discussion_avg,
  'name': 'discussion average',
  'type': 'bar'
};
df = [trace1, trace2];
layout = {
  'xaxis': {'title': 'PlaceofBirth'},
  'barmode': 'relative',
  'title': 'Raisedhands and Discussion Average Acording to Place of Birth'
};
fig = go.Figure(data = df, layout = layout)
iplot(fig)
# Raisedhands vs  Discussion Rate point plot
#normalize the values of discussion_avg and raisedhands_avg
data3=sorted_data2.copy()
data3["raisedhands_avg"]=data3['raisedhands_avg']/max( data3['raisedhands_avg'])
data3["discussion_avg"]=data3['discussion_avg']/max( data3['discussion_avg'])

# visualize
f,ax1 = plt.subplots(figsize =(12,10))
sns.pointplot(x='topic',y='raisedhands_avg',data=data3,color='lime',alpha=0.8)
sns.pointplot(x='topic',y='discussion_avg',data=data3,color='red',alpha=0.8)
plt.text(5,0.50,'Raised hands Average',color='red',fontsize = 17,style = 'italic')
plt.text(5,0.46,'Discussion Average',color='lime',fontsize = 18,style = 'italic')
plt.xlabel('Topics',fontsize = 15,color='blue')
plt.ylabel('Values',fontsize = 15,color='blue')
plt.title('Raisedhands vs  Discussion Rate',fontsize = 20,color='blue')
plt.grid()



# Raisedhands vs  Discussion Rate point plot acording to place of birth
#normalize the values of discussion_avg and raisedhands_avg
data5=sorted_data4.copy()
data5["raisedhands_avg"]=data5['raisedhands_avg']/max( data5['raisedhands_avg'])
data5["discussion_avg"]=data5['discussion_avg']/max( data5['discussion_avg'])

# visualize
f,ax1 = plt.subplots(figsize =(12,10))
sns.pointplot(x='PlaceofBirth',y='raisedhands_avg',data=data5,color='red',alpha=0.8)
sns.pointplot(x='PlaceofBirth',y='discussion_avg',data=data5,color='blue',alpha=0.8)
plt.text(3,0.30,'Raised hands Average',color='red',fontsize = 17,style = 'italic')
plt.text(3,0.36,'Discussion Average',color='blue',fontsize = 18,style = 'italic')
plt.xlabel('PlaceofBirth',fontsize = 15,color='purple')
plt.ylabel('Values',fontsize = 15,color='purple')
plt.title('Raisedhands vs  Discussion Rate',fontsize = 20,color='purple')
plt.grid()
data.gender.value_counts()
plt.subplots(figsize=(8,5))
sns.countplot(data.gender)
plt.xlabel("gender",fontsize="15")
plt.ylabel("numbers",fontsize="15")
plt.title("Number of Genders in Data", color="red",fontsize="18")
plt.show()
#StageID unique values
data.StageID.value_counts()
sns.countplot(data.StageID)
plt.xlabel("StageID")
plt.ylabel("numbers")
plt.title("Number of StageID in Data", color="red",fontsize="18")
plt.show()
labels=data.StageID.value_counts()
colors=["grey","blue","green"]
explode=[0,0,0]
sizes=data.StageID.value_counts().values

plt.figure(figsize = (7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title("StageID in Data",color = 'blue',fontsize = 15)
plt.show()
# StageID piechart in plotly
pie1_list=data["StageID"].value_counts().values
labels = data["StageID"].value_counts().index
# figure
fig = {
  "data": [
    {
      "values": pie1_list,
      "labels": labels,
      "domain": {"x": [0, .5]},
      "name": "StageID",
      "hoverinfo":"label+percent",
      "hole": .3,
      "type": "pie"
    },],
  "layout": {
        "title":"StageID Type",
        "annotations": [
            { "font": { "size": 20},
              "showarrow": False,
              "text": "StageID",
                "x": 0.20,
                "y": 1
            },
        ]
    }
}
iplot(fig)
data.head()
# raisedhands and VisITedResources pair plot
sns.pairplot(data.loc[:,["raisedhands","VisITedResources","Discussion"]])
plt.show()
data.head()
data_new=data.loc[:,["gender","raisedhands","VisITedResources","AnnouncementsView","Discussion"]]
data_new.gender=[1 if i=="M" else 0 for i in data_new.gender]
data_new.head()
y=data_new.gender.values
x_data=data_new.drop("gender",axis=1)
# normalize the values in x_data
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=52)
#Logistic Regression
from sklearn.linear_model import LogisticRegression
#fit
lr=LogisticRegression()
lr.fit(x_train,y_train)

#accuracy
print("test accuracy is {}".format(lr.score(x_test,y_test)))

#split data
from sklearn.neighbors import KNeighborsClassifier
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

knn=KNeighborsClassifier(n_neighbors=3)

#fit
knn.fit(x_train,y_train)

#prediction
prediction=knn.predict(x_test)
#prediction score (accuracy)
print('KNN (K=3) accuracy is: ',knn.score(x_test,y_test)) 
# find the convenient k value for range (1,31)
score_list=[]
train_accuracy=[]
for i in range(1,31):
    knn2=KNeighborsClassifier(n_neighbors=i)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
    train_accuracy.append(knn2.score(x_train,y_train))
plt.figure(figsize=(15,10))   
plt.plot(range(1,31),score_list,label="testing accuracy",color="blue",linewidth=3)
plt.plot(range(1,31),train_accuracy,label="training accuracy",color="orange",linewidth=3)
plt.xlabel("k values in KNN")
plt.ylabel("accuracy")
plt.title("Accuracy results with respect to k values")
plt.legend()
plt.grid()
plt.show()

print("Maximum value of testing accuracy is {} when k= {}.".format(np.max(score_list),1+score_list.index(np.max(score_list))))
from sklearn.svm import SVC

svm=SVC(random_state=1)
svm.fit(x_train,y_train)
#accuracy
print("accuracy of svm algorithm: ",svm.score(x_test,y_test))

from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train,y_train)

# test accuracy
print("Accuracy of naive bayees algorithm: ",nb.score(x_test,y_test))

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)

print("Accuracy score for Decision Tree Classification: " ,dt.score(x_test,y_test))

from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=100,random_state=1)
rf.fit(x_train,y_train)

print("random forest algorithm accuracy: ",rf.score(x_test,y_test))

score_list1=[]
for i in range(100,501,100):
    rf2=RandomForestClassifier(n_estimators=i,random_state=1)
    rf2.fit(x_train,y_train)
    score_list1.append(rf2.score(x_test,y_test))
plt.figure(figsize=(10,10))
plt.plot(range(100,501,100),score_list1)
plt.xlabel("number of estimators")
plt.ylabel("accuracy")
plt.grid()
plt.show()

print("Maximum value of accuracy is {} \nwhen n_estimators= {}.".format(max(score_list1),(1+score_list1.index(max(score_list1)))*100))
score_list2=[]
for i in range(100,131):
    rf3=RandomForestClassifier(n_estimators=i,random_state=1)
    rf3.fit(x_train,y_train)
    score_list2.append(rf3.score(x_test,y_test))
plt.figure(figsize=(10,10))
plt.plot(range(100,131),score_list2)
plt.xlabel("number of estimators")
plt.ylabel("accuracy")
plt.grid()
plt.show()

print("Maximum value of accuracy is {} when number of estimators between 100 and 131 ".format(max(score_list2)))
#Confusion matrix of Random Forest Classf.
y_pred=rf.predict(x_test)
y_true=y_test

#cm
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_true,y_pred)

#cm visualization
f,ax=plt.subplots(figsize=(8,8))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("predicted value")
plt.ylabel("real value")
plt.show()
#Confusion matrix of KNN Classf.
y_pred1=knn.predict(x_test)
y_true=y_test
#cm
cm1=confusion_matrix(y_true,y_pred1)

#cm visualization
f,ax=plt.subplots(figsize=(8,8))
sns.heatmap(cm1,annot=True,linewidths=0.5,linecolor="blue",fmt=".0f",ax=ax)
plt.xlabel("predicted value")
plt.ylabel("real value")
plt.show()
#Confusion matrix of Decision Tree Classf.
y_pred2=dt.predict(x_test)
y_true=y_test
#cm
cm2=confusion_matrix(y_true,y_pred2)

#cm visualization
f,ax=plt.subplots(figsize=(8,8))
sns.heatmap(cm2,annot=True,linewidths=0.5,linecolor="green",fmt=".0f",ax=ax)
plt.xlabel("predicted value")
plt.ylabel("real value")
plt.show()
dictionary={"model":["LR","KNN","SVM","NB","DT","RF"],"score":[lr.score(x_test,y_test),knn.score(x_test,y_test),svm.score(x_test,y_test),nb.score(x_test,y_test),dt.score(x_test,y_test),rf.score(x_test,y_test)]}
df1=pd.DataFrame(dictionary)
#sort the values of data 
new_index5=df1.score.sort_values(ascending=False).index.values
sorted_data5=df1.reindex(new_index5)

# create trace1 
trace1 = go.Bar(
                x = sorted_data5.model,
                y = sorted_data5.score,
                name = "score",
                marker = dict(color = 'rgba(200, 125, 200, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = sorted_data5.model)
dat = [trace1]
layout = go.Layout(barmode = "group",title= 'Scores of Classifications')
fig = go.Figure(data = dat, layout = layout)
iplot(fig)