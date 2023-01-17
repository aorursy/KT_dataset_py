import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics # for spliting dataset
#importing classification algorithms
from sklearn.tree import DecisionTreeClassifier
# from sklearn import svm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import os
# print

data = pd.read_csv("../input/Iris.csv") #loading the dataset

data.info() #checking if there are any null values
#in our dataset there is no inconsistency(NULL values)
# so we can work on it

data.drop('Id', axis=1, inplace = True) #dropping unecessary column
#for plotting Sepal Length vs Sepal Width
'''
fig = data[data.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='red',label='Setosa')
data[data.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green',label='Virginica' ,ax=fig)
data[data.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue',label='Versicolor',ax=fig)

fig.set_xlabel("Sepal Length")
fig.set_ylabel("Sepal Width")
fig.set_title("Sepal Length vs Sepal Width")
fig = plt.gcf()
fig.set_size_inches(10,6)
plt.show()
'''
#for plotting petal length vs petal width
'''
fig = data[data.Species=='Iris-setosa'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='red',label='Setosa')
data[data.Species=='Iris-virginica'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='green',label='Virginica' ,ax=fig)
data[data.Species=='Iris-versicolor'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='blue',label='Versicolor',ax=fig)

fig.set_xlabel("Petal Length")
fig.set_ylabel("Petal Width")
fig.set_title("Petal Length vs Petal Width")
fig = plt.gcf()
fig.set_size_inches(10,6)
plt.show()
'''

#splitting the dataset into Training and test set of 70/30 ratio with random state off and stratify the column species 
#stratify spit the dataset into some proportion into training and test set
train, test = train_test_split(data,test_size = 0.3,random_state=0,stratify=data['Species'])

print(train.shape,test.shape) #printing size of train and tes set

trainX = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']] #training features
trainY = train[['Species']]	# output data train

testX = test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']] #testing features
testY = test[['Species']]  # output data test

models = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression()
]
i=0
tab=[]
classifiers=["KNeighbor","SVC","SVC2","DecisionTree","RandomForest","LogisticRegression"]
for model in models:
	model.fit(trainX,trainY)							#train algorithm with raining data and training output
	y_predict = model.predict(testX)					#test the trained algorithm on testing data
	tab.append(metrics.accuracy_score(y_predict,testY))	#checking our predicted output with real output and adding it to array
table = pd.DataFrame(tab,index=classifiers)			 	#creating table with index as classifiers
table.columns=['Accuracy']
print(table)
