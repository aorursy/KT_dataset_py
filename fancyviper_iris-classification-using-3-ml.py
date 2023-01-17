import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import plotly.offline as pyo
import cufflinks as cf
from plotly.offline import init_notebook_mode,plot,iplot

import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.metrics import accuracy_score
import os
pyo.init_notebook_mode(connected=True)
cf.go_offline()
iris=pd.read_csv("../input/iris/Iris.csv")
iris
iris.shape
iris.drop('Id',axis=1,inplace=True)
iris
px.scatter(iris,x='Species',y='PetalWidthCm')
px.scatter(iris,x='Species',y='PetalWidthCm',size='PetalWidthCm')
plt.bar(iris['Species'],iris['PetalWidthCm'])##matplotlib
##plotly express
px.bar(iris,x='Species',y='PetalWidthCm')
iris.iplot(kind='bar',x=['Species'],y=['PetalWidthCm'])
px.line(iris,x='Species',y='PetalWidthCm')
iris.rename(columns={'SepalLengthCm':'SepalLength','SepalWidthCm':'SepalWidth','PetalLengthCm':'PetalLength','PetalWidthCm':'PetalWidth'},inplace=True)
iris
px.scatter_matrix(iris,color='Species',title='Iris',dimensions=['SepalLength','SepalWidth','PetalWidth','PetalLength'])
iris
X=iris.drop(['Species'],axis=1)
X
y=iris['Species']
y
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

y=le.fit_transform(y)
y
X=np.array(X)
from sklearn.model_selection import train_test_split  ##(train and predict,x-features,y-labels)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0) ## test size 30% of data  random state=int (return to same value) or return to random value 

X_train.size

X_train
y_test
X_test.size

from sklearn import tree
dt=tree.DecisionTreeClassifier()
dt.fit(X_train,y_train)
predict_dt=dt.predict(X_test)
accuracy_dt=accuracy_score(y_test,predict_dt)*100
accuracy_dt
y_test
predict_dt
os.environ["PATH"]+= os.pathsep+(r'C:\Program Files (x86)\Graphviz2.38\bin')
import graphviz
vis_data=tree.export_graphviz(dt,out_file=None,feature_names=iris.drop(['Species'],axis=1).keys(),class_names=iris['Species'].unique(),filled=True,rounded=True, special_characters=True,)
graphviz.Source(vis_data)
iris['Species'].unique()
Category=['Iris-setosa','Iris-versicolor','Iris-virginica']
x_Data=5,2.3,3.3
x_Data
x_Dt=np.array([[1, 1, 1,1]])
x_Dt_prediction=dt.predict(x_Dt)
x_Dt_prediction[0]
print(Category[int(x_Dt_prediction[0])])
iris.head(3)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler().fit(X_train) ## analyze pattern ,cm,inch,km,all units
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)

X_train[0:3,:]
X_train_std[0:3,:]
from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_std,y_train)
predict_knn=knn.predict(X_test_std)
accuracy_knn=accuracy_score(y_test,predict_knn)*100
accuracy_knn
x_knn=np.array([[6, 3.3, 6, 2.5]])

x_knn_std=sc.transform(x_knn)
x_knn_std
x_knn_prediction=knn.predict(x_knn_std)
x_knn_prediction[0]
print(Category[int(x_knn_prediction[0])])
k_range=range(1,26)
scores={}
scores_list=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_std,y_train)
    prediction_knn=knn.predict(X_test_std)
    scores[k]=accuracy_score(y_test,prediction_knn)
    scores_list.append(accuracy_score(y_test,prediction_knn))
scores_list
plt.plot(k_range,scores_list)
y
colormap=np.array(['Red','green','blue'])
fig=plt.scatter(iris['PetalLength'],iris['PetalWidth'],c=colormap[y],s=50)
iris
X
from sklearn.cluster import KMeans
kn=KMeans(n_clusters=3,random_state=2,n_jobs=4) ##n_jobs =paralization
kn.fit(X)


centers=kn.cluster_centers_
print(centers)
kn.labels_

Category
Category=['Iris-versicolor','Iris-setosa','Iris-virginica']
Category
colormap=np.array(['Red','green','blue'])
fig=plt.scatter(iris['PetalLength'],iris['PetalWidth'],c=colormap[kn.labels_],s=50)

new_labels=kn.labels_
fig,axes=plt.subplots(1,2,figsize=(16,8))
axes[0].scatter(X[:,2],X[:,3],c=y,cmap='gist_rainbow',edgecolor='k',s=150)
axes[1].scatter(X[:,2],X[:,3],c=y,cmap='jet',edgecolor='k',s=150)
axes[0].set_title('actual',fontsize=18)
axes[1].set_title('predicted',fontsize=18)
x_kn=np.array([[6, 3, 4.8 ,1.8]])

x_kn_prediction=kn.predict(x_kn)
x_kn_prediction[0]
print(Category[int(x_kn_prediction[0])])
