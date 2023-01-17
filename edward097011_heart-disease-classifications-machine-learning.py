# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# We are reading our data

df = pd.read_csv("../input/heart.csv")
# First 5 rows of our data

df.head()
df.target.value_counts()
countNoDisease = len(df[df.target == 0])

countHaveDisease = len(df[df.target == 1])

print("Percentage of Patients Haven't Heart Disease: {:.2f}%".format((countNoDisease / (len(df.target))*100)))

print("Percentage of Patients Have Heart Disease: {:.2f}%".format((countHaveDisease / (len(df.target))*100)))
countFemale = len(df[df.sex == 0])

countMale = len(df[df.sex == 1])

print("Percentage of Female Patients: {:.2f}%".format((countFemale / (len(df.sex))*100)))

print("Percentage of Male Patients: {:.2f}%".format((countMale / (len(df.sex))*100)))
df.groupby('target').mean()
plt.scatter(x=df.age[df.target==1], y=df.thalach[(df.target==1)], c="red")

plt.scatter(x=df.age[df.target==0], y=df.thalach[(df.target==0)])

plt.legend(["Disease", "Not Disease"])

plt.xlabel("Age")

plt.ylabel("Maximum Heart Rate")

plt.show()
a = pd.get_dummies(df['cp'], prefix = "cp")

b = pd.get_dummies(df['thal'], prefix = "thal")

c = pd.get_dummies(df['slope'], prefix = "slope")
frames = [df, a, b, c]

df = pd.concat(frames, axis = 1)

df.head()
df = df.drop(columns = ['cp', 'thal', 'slope'])

df.head()
y = df.target.values

x_data = df.drop(['target'], axis = 1)
# Normalize

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
#transpose matrices

x_train = x_train.T

y_train = y_train.T

x_test = x_test.T

y_test = y_test.T
#initialize

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

    

    print("Manuel Test Accuracy: {:.2f}%".format((100 - np.mean(np.abs(y_prediction - y_test))*100)))
logistic_regression(x_train,y_train,x_test,y_test,1,100)
from sklearn.svm import SVC
svm = SVC(random_state = 1)

svm.fit(x_train.T, y_train.T)



acc = svm.score(x_test.T,y_test.T)*100

accuracies['SVM'] = acc

print("Test Accuracy of SVM Algorithm: {:.2f}%".format(acc))
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train.T, y_train.T)



acc = nb.score(x_test.T,y_test.T)*100

accuracies['Naive Bayes'] = acc

print("Accuracy of Naive Bayes: {:.2f}%".format(acc))
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

dtc.fit(x_train.T, y_train.T)



acc = dtc.score(x_test.T, y_test.T)*100

accuracies['Decision Tree'] = acc

print("Decision Tree Test Accuracy {:.2f}%".format(acc))
# Random Forest Classification

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)

rf.fit(x_train.T, y_train.T)



acc = rf.score(x_test.T,y_test.T)*100

accuracies['Random Forest'] = acc

print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(acc))
# Predicted values

y_head_lr = lr.predict(x_test.T)

knn3 = KNeighborsClassifier(n_neighbors = 3)

knn3.fit(x_train.T, y_train.T)

y_head_knn = knn3.predict(x_test.T)

y_head_svm = svm.predict(x_test.T)

y_head_nb = nb.predict(x_test.T)

y_head_dtc = dtc.predict(x_test.T)

y_head_rf = rf.predict(x_test.T)
from sklearn.metrics import confusion_matrix



cm_lr = confusion_matrix(y_test,y_head_lr)

cm_knn = confusion_matrix(y_test,y_head_knn)

cm_svm = confusion_matrix(y_test,y_head_svm)

cm_nb = confusion_matrix(y_test,y_head_nb)

cm_dtc = confusion_matrix(y_test,y_head_dtc)

cm_rf = confusion_matrix(y_test,y_head_rf)

plt.figure(figsize=(24,12))



plt.suptitle("Confusion Matrixes",fontsize=24)

plt.subplots_adjust(wspace = 0.4, hspace= 0.4)



plt.subplot(2,3,1)

plt.title("Logistic Regression Confusion Matrix")

sns.heatmap(cm_lr,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(2,3,2)

plt.title("K Nearest Neighbors Confusion Matrix")

sns.heatmap(cm_knn,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(2,3,3)

plt.title("Support Vector Machine Confusion Matrix")

sns.heatmap(cm_svm,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(2,3,4)

plt.title("Naive Bayes Confusion Matrix")

sns.heatmap(cm_nb,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(2,3,5)

plt.title("Decision Tree Classifier Confusion Matrix")

sns.heatmap(cm_dtc,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(2,3,6)

plt.title("Random Forest Confusion Matrix")

sns.heatmap(cm_rf,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.show()