# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB



import os

print(os.listdir("../input"))



data =pd.read_csv("../input/heart.csv")
data.info()
data.head()
data.columns
print(data['target'].value_counts(dropna =False))
data.describe()
data.corr()
f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

outlierlower_thalach = 133.5-1.5*(166-133.5)

print('outlierlower_thalach =',outlierlower_thalach)



outlierupper_thalach = 166+1.5*(166-133.5)

print('outlierupper_thalach =',outlierupper_thalach)
data.boxplot(column='thalach',by = 'target')

data[(data['target']==1)].info()
data[(data['target']==1)].describe()
outlierlower_thalach1 = 149-1.5*(172-149)

print('outlierlower_thalach1 =',outlierlower_thalach1)



outlierupper_thalach1 = 172+1.5*(172-149)

print('outlierupper_thalach1 =',outlierupper_thalach1)
data[(data['thalach']<114.5) & (data['target']==1)].head()
data['thalach'] = data['thalach'].astype('float')
data.drop([17,95,136,139], axis=0, inplace=True)
data.info()
data[(data['target']==1)].describe()
outlierlower_thalach2 = 159-1.5*(172-150)

print('outlierlower_thalach2 =',outlierlower_thalach2)



outlierupper_thalach2 = 172+1.5*(172-150)

print('outlierupper_thalach2 =',outlierupper_thalach2)
data.boxplot(column='thalach',by = 'target')
data.boxplot(column='cp',by = 'target')
data.boxplot(column='slope',by = 'target')
data.plot(kind='scatter', x='thalach', y='slope',alpha = 0.5,color = 'red')

plt.xlabel('thalach')              # label = name of label

plt.ylabel('slope')

plt.title('thalach slope Scatter Plot')  
data.plot(kind='scatter', x='target', y='cp',alpha = 0.5,color = 'red')

plt.xlabel('target')              # label = name of label

plt.ylabel('cp')

plt.title('target cp Scatter Plot')  
data[(data['slope']>=1) & (data['thalach']>100) & (data['target']==1)]
data1= data[(data['slope']>=1) & (data['thalach']>100) & (data['target']==1)].head()

data2= data[(data['slope']>=1) & (data['thalach']>100) & (data['target']==1)].tail()

conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =False) # axis = 0 : adds dataframes in row

conc_data_row
data1 = data[(data['slope']>=1) & (data['thalach']>100) & (data['target']==1)]['age']

data2= data[(data['slope']>=1) & (data['thalach']>100) & (data['target']==1)]['sex']

conc_data_col = pd.concat([data1,data2],axis =1) # axis = 0 : adds dataframes in row

conc_data_col
data[(data['slope']>=1) & (data['thalach']>100) & (data['target']==1)].info()
fig, axes = plt.subplots(nrows=2,ncols=1)

data[(data['slope']>=1) & (data['thalach']>100) & (data['target']==1)].plot(kind = "hist",y = "cp",bins = 50,normed = True,ax = axes[0])

data[(data['slope']>=1) & (data['thalach']>100) & (data['target']==1)].plot(kind = "hist",y = "cp",bins = 50,normed = True,ax = axes[1],cumulative = True)

plt.savefig('graph.png')

plt
fig, axes = plt.subplots(nrows=2,ncols=1)

data[(data['slope']>=1) & (data['thalach']>100) & (data['target']==1)].plot(kind = "hist",y = "slope",bins = 50,normed = True,ax = axes[0])

data[(data['slope']>=1) & (data['thalach']>100) & (data['target']==1)].plot(kind = "hist",y = "slope",bins = 50,normed = True,ax = axes[1],cumulative = True)

plt.savefig('graph.png')

plt
fig, axes = plt.subplots(nrows=2,ncols=1)

data[(data['slope']>=1) & (data['thalach']>100) & (data['target']==1)].plot(kind = "hist",y = "thalach",bins = 50,normed = True,ax = axes[0])

data[(data['slope']>=1) & (data['thalach']>100) & (data['target']==1)].plot(kind = "hist",y = "thalach",bins = 50,normed = True,ax = axes[1],cumulative = True)

plt.savefig('graph.png')

plt
y = data.target.values

x_data = data.drop(['target'], axis = 1)
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
x_train = x_train.T

y_train = y_train.T

x_test = x_test.T

y_test = y_test.T

knn = KNeighborsClassifier(n_neighbors = 2)  # n_neighbors means k

knn.fit(x_train.T, y_train.T)

prediction = knn.predict(x_test.T)



print("{} NN Score: {:.2f}%".format(2, knn.score(x_test.T, y_test.T)*100))
scoreList = []

for i in range(1,20):

    knn2 = KNeighborsClassifier(n_neighbors = i)  # n_neighbors means k

    knn2.fit(x_train.T, y_train.T)

    scoreList.append(knn2.score(x_test.T, y_test.T))

    

plt.plot(range(1,20), scoreList)

plt.xticks(np.arange(1,20,1))

plt.xlabel("K value")

plt.ylabel("Score")

plt.show()





print("Maximum KNN Score is {:.2f}%".format((max(scoreList))*100))
svm = SVC(random_state = 1)

svm.fit(x_train.T, y_train.T)
print("Test Accuracy of SVM Algorithm: {:.2f}%".format(svm.score(x_test.T,y_test.T)*100))
nb = GaussianNB()

nb.fit(x_train.T, y_train.T)

print("Accuracy of Naive Bayes: {:.2f}%".format(nb.score(x_test.T,y_test.T)*100))
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

dtc.fit(x_train.T, y_train.T)

print("Decision Tree Test Accuracy {:.2f}%".format(dtc.score(x_test.T, y_test.T)*100))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)

rf.fit(x_train.T, y_train.T)

print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(rf.score(x_test.T,y_test.T)*100))
def initialize(dimension):

    

    weight = np.full((dimension,1),0.01)

    bias = 0.0

    return weight,bias
def sigmoid(z):

    

    y_head = 1/(1+ np.exp(-z))

    return y_head
def forwardBackward(weight,bias,x_train,y_train):

    # Forward

    

    y_head = sigmoid(np.dot(weight.T,x_train) + bias)

    loss = -(y_train*np.log(y_head) + (1-y_train)*np.log(1-y_head))

    cost = np.sum(loss) / x_train.shape[1]

    

    # Backward

    derivative_weight = np.dot(x_train,((y_head-y_train).T))/x_train.shape[1]

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]

    gradients = {"Derivative Weight" : derivative_weight, "Derivative Bias" : derivative_bias}

    

    return cost,gradients
def update(weight,bias,x_train,y_train,learningRate,iteration) :

    costList = []

    index = []

    

    #for each iteration, update weight and bias values

    for i in range(iteration):

        cost,gradients = forwardBackward(weight,bias,x_train,y_train)

        weight = weight - learningRate * gradients["Derivative Weight"]

        bias = bias - learningRate * gradients["Derivative Bias"]

        

        costList.append(cost)

        index.append(i)



    parameters = {"weight": weight,"bias": bias}

    

    print("iteration:",iteration)

    print("cost:",cost)



    plt.plot(index,costList)

    plt.xlabel("Number of Iteration")

    plt.ylabel("Cost")

    plt.show()



    return parameters, gradients
def predict(weight,bias,x_test):

    z = np.dot(weight.T,x_test) + bias

    y_head = sigmoid(z)



    y_prediction = np.zeros((1,x_test.shape[1]))

    

    for i in range(y_head.shape[1]):

        if y_head[0,i] <= 0.5:

            y_prediction[0,i] = 0

        else:

            y_prediction[0,i] = 1

    return y_prediction
def logistic_regression(x_train,y_train,x_test,y_test,learningRate,iteration):

    dimension = x_train.shape[0]

    weight,bias = initialize(dimension)

    

    parameters, gradients = update(weight,bias,x_train,y_train,learningRate,iteration)



    y_prediction = predict(parameters["weight"],parameters["bias"],x_test)

    



    print("Manuel Test Accuracy: {:.2f}%".format((100 - np.mean(np.abs(y_prediction - y_test))*100)/100*100))
logistic_regression(x_train,y_train,x_test,y_test,1,100)
methods = ["Logistic Regression", "KNN", "SVM", "Naive Bayes", "Decision Tree", "Random Forest"]

accuracy = [86.89, 88.52, 86.89, 86.89, 78.69, 88.52]

colors = ["purple", "green", "orange", "magenta","#CFC60E","#0FBBAE"]



sns.set_style("whitegrid")

plt.figure(figsize=(16,5))

plt.yticks(np.arange(0,100,10))

plt.ylabel("Accuracy %")

plt.xlabel("Algorithms")

sns.barplot(x=methods, y=accuracy, palette=colors)

plt.show()