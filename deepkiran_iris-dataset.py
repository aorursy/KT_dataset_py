

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import os

import matplotlib.pyplot as plt

from subprocess import check_output

print(os.listdir("../input"))

print(check_output(["ls", "../input"]).decode("utf8"))



iris = pd.read_csv("../input/Iris.csv") #load the dataset

iris.head(5)
iris.info()
iris.drop('Id',axis=1,inplace =True) #drop Id from column which also reflects in data frame
fig =iris[iris.Species=='Iris-setosa'].plot(kind ='scatter',x='SepalLengthCm',y ='SepalWidthCm',color ='orange', 

                                           label ='Setosa')

iris[iris.Species=='Iris-versicolor'].plot(kind ='scatter',x='SepalLengthCm',y ='SepalWidthCm',color ='blue', 

                                           label ='Versicolor',ax= fig)

iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', 

                                          label='Virginica', ax=fig)

fig.set_title("Sepal Length VS Width")

fig.set_xlabel('Sepal Length')

fig.set_ylabel('Sepal Width')

fig=plt.gcf() #get current figure

fig.set_size_inches(10,6) #enlarges the graph

plt.show()
fig =iris[iris.Species=='Iris-setosa'].plot(kind ='scatter',x='PetalLengthCm',y ='PetalWidthCm',color ='orange', 

                                           label ='Setosa')

iris[iris.Species=='Iris-versicolor'].plot(kind ='scatter',x='PetalLengthCm',y ='PetalWidthCm',color ='blue', 

                                           label ='Versicolor',ax= fig)

iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='green', 

                                          label='Virginica', ax=fig)

fig.set_title("Petal Length VS Width")

fig.set_xlabel('Petal Length')

fig.set_ylabel('Petal Width')

fig=plt.gcf() #get current figure

fig.set_size_inches(10,6) #enlarges the graph

plt.show()
iris.hist(edgeColor='black',linewidth=1.2)

fig=plt.gcf()

fig.set_size_inches(12,6)

plt.show()
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.violinplot(x='Species',y='SepalLengthCm',data=iris)

plt.subplot(2,2,2)

sns.violinplot(x='Species',y='SepalWidthCm',data=iris)

plt.subplot(2,2,3)

sns.violinplot(x='Species',y='PetalLengthCm',data=iris)

plt.subplot(2,2,4)

sns.violinplot(x='Species',y='PetalWidthCm',data =iris)
from sklearn.linear_model import LogisticRegression

#from sklearn.cross_validation import train_test_split ---- this is depreciated

from sklearn.model_selection import train_test_split #to split the dataset for training and testing

from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm

from sklearn import metrics #for checking model accuracy

from sklearn.tree import DecisionTreeClassifier
iris.shape
plt.figure(figsize=(7,5))

sns.heatmap(data =iris.corr(),annot=True,cmap='cubehelix_r')
train, test =train_test_split(iris, test_size =0.3)

print(train.shape)

print(test.shape)
train_x =train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

train_y =train[['Species']]

test_x =test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

test_y =test[['Species']]
model =svm.SVC()

model.fit(train_x,train_y)

prediction =model.predict(test_x)

print("The accuracy of the SVM model is",metrics.accuracy_score(prediction,test_y))
model =LogisticRegression()

model.fit(train_x,train_y)

predictions =model.predict(test_x)

print("The accuracy with Logistic Regression is ",metrics.accuracy_score(predictions,test_y))
model=KNeighborsClassifier(n_neighbors=3) #this examines 3 neighbours for putting the new data into a class

model.fit(train_x,train_y)

prediction=model.predict(test_x)

print('The accuracy of the KNN is',metrics.accuracy_score(prediction,test_y))
a_index=list(range(1,11))

a=pd.Series()

x=[1,2,3,4,5,6,7,8,9,10]

for i in list(range(1,11)):    

    model=KNeighborsClassifier(n_neighbors=i) 

    model.fit(train_x,train_y)

    prediction=model.predict(test_x)

    a=a.append(pd.Series(metrics.accuracy_score(prediction,test_y)))

plt.plot(a_index, a)

plt.xticks(x) #marks x-axis 
petal=iris[['PetalLengthCm','PetalWidthCm','Species']]

sepal=iris[['SepalLengthCm','SepalWidthCm','Species']]
train_p,test_p=train_test_split(petal,test_size=0.3,random_state=0)  #petals

train_x_p=train_p[['PetalWidthCm','PetalLengthCm']]

train_y_p=train_p[['Species']]

test_x_p=test_p[['PetalWidthCm','PetalLengthCm']]

test_y_p=test_p[['Species']]





train_s,test_s=train_test_split(sepal,test_size=0.3,random_state=0)  #Sepal

train_x_s=train_s[['SepalWidthCm','SepalLengthCm']]

train_y_s=train_s.Species

test_x_s=test_s[['SepalWidthCm','SepalLengthCm']]

test_y_s=test_s.Species
model =svm.SVC()

model.fit(train_x_p,train_y_p)

prediction =model.predict(test_x_p)

print("The accuracy of SVM model with petals as features are",metrics.accuracy_score(prediction,test_y_p))



model =svm.SVC()

model.fit(train_x_s,train_y_s)

prediction =model.predict(test_x_s)

print("The accuracy of SVM model with sepals as features are",metrics.accuracy_score(prediction,test_y_p))
model = LogisticRegression()

model.fit(train_x_p,train_y_p) 

prediction=model.predict(test_x_p) 

print('The accuracy of the Logistic Regression using Petals is:',metrics.accuracy_score(prediction,test_y_p))



model.fit(train_x_s,train_y_s) 

prediction=model.predict(test_x_s) 

print('The accuracy of the Logistic Regression using Sepals is:',metrics.accuracy_score(prediction,test_y_s))
model=DecisionTreeClassifier()

model.fit(train_x_p,train_y_p) 

prediction=model.predict(test_x_p) 

print('The accuracy of the Decision Tree using Petals is:',metrics.accuracy_score(prediction,test_y_p))



model.fit(train_x_s,train_y_s) 

prediction=model.predict(test_x_s) 

print('The accuracy of the Decision Tree using Sepals is:',metrics.accuracy_score(prediction,test_y_s))
model=KNeighborsClassifier(n_neighbors=3) 

model.fit(train_x_p,train_y_p) 

prediction=model.predict(test_x_p) 

print('The accuracy of the KNN using Petals is:',metrics.accuracy_score(prediction,test_y_p))



model.fit(train_x_s,train_y_s) 

prediction=model.predict(test_x_s) 

print('The accuracy of the KNN using Sepals is:',metrics.accuracy_score(prediction,test_y_s))