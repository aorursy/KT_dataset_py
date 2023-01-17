# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#ReadData

data = pd.read_csv("../input/heart.csv")
data.info()
data.head()
#Seperate data

y =data.target.values

x1=data.drop(["target"],axis=1)
#Normalization 

x = (x1 - np.min(x1))/(np.max(x1)-np.min(x1)).values
#Split For Train and Test

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2, random_state=42)
#transposition

xtrain = xtrain.T

xtest = xtest.T

ytrain = ytrain.T

ytest = ytest.T
#Initializing Parametres & Sigmoid Function

def initializing_weights_and_bias(dimension):

    w = np.full((dimension,1),0.01)

    b = 0.0

    return w,b

def sigmoid(z):

    y_head = 1/(1+ np.exp(-z))

    return y_head
#Forward & Backward Propogation



def forward_backwardpropogation(w,b,xtrain,y_train):

    #forward P.

    z = np.dot(w.T,xtrain) +  b 

    y_head = sigmoid(z)

    loss = -ytrain*np.log(y_head)-(1-ytrain)*np.log(1-y_head)

    cost = (np.sum(loss))/xtrain.shape[1]

    #backward p.

    derivative_weight = (np.dot(xtrain,((y_head-ytrain).T)))/xtrain.shape[1] 

    derivetive_bias = np.sum(y_head-ytrain)/xtrain.shape[1]

    gradients = {"derivative_weight": derivative_weight,"derivetive_bias": derivetive_bias}

    

    return cost,gradients
#Uptading parameters



def update(w,b,xtrain, ytrain, learning_rate, number_of_iteration):

    costlist = []

    costlist2 = []

    index = []

    #updating/learning parameters is number of iteration times

    for i in range(number_of_iteration):

        #makeforwardandbacwardprop.andfindcostandgradi

        cost,gradients = forward_backwardpropogation(w,b,xtrain,ytrain)

        costlist.append(cost)

        #updatingtime

        w = w - learning_rate*gradients["derivative_weight"]

        b = b - learning_rate*gradients["derivetive_bias"]

        if i % 10 == 0:

            costlist2.append(cost)

            index.append(i)

            print("Cost after iteration %i: %f" %(i,cost))

            

    #weupdate(learn) parameters weight & bias

    parameters = {"weight": w, "bias":b}

    plt.plot(index,costlist2)

    plt.xticks(index,rotation='vertical')

    plt.xlabel("Number of iteration")

    plt.ylabel("Cost")

    plt.show()

    return parameters, gradients, costlist
#%% Prediciton Method

    

def predict(w,b,xtest):

    

    z =sigmoid(np.dot(w.T,xtest)+b)

    y_prediction = np.zeros((1,xtest.shape[1]))

    #if z > 0.5 predcition = 1 y_head=1

    #if z <= 0.5 prediciton = 0 y_head=0

    for i in range(z.shape[1]):

        if z[0,i]<=0.5:

            y_prediction[0,i] = 0

        else:

            y_prediction[0,i] = 1

    return y_prediction
# Logistic Reg.



def LogReg(xtrain,ytrain,xtest,ytest,learning_rate,number_of_iteration):

    #initializing

    dimension = xtrain.shape[0] 

    w,b=initializing_weights_and_bias(dimension)

    #forward & backward prop.

    parameters,gradients,costlist = update(w,b,xtrain,ytrain,learning_rate,number_of_iteration)

    

    y_prediciton_test = predict(parameters["weight"],parameters["bias"],xtest)



    #print train/test errors

    print("Test Accuracy:{} %".format(100-np.mean(np.abs(y_prediciton_test-ytest))*100))



#Application1

LogReg(xtrain,ytrain,xtest,ytest,learning_rate=1,number_of_iteration =50 )
#Application2

LogReg(xtrain,ytrain,xtest,ytest,learning_rate=1,number_of_iteration =100 )
#Application3

LogReg(xtrain,ytrain,xtest,ytest,learning_rate=1,number_of_iteration =1000)
#Application4

LogReg(xtrain,ytrain,xtest,ytest,learning_rate=3,number_of_iteration =3000)
#LR with sklearn

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()

LR.fit(xtrain.T,ytrain.T)

print("Test Accuracy {}".format(LR.score(xtest.T,ytest.T))) 
#Confusion Matrix



yprediciton= LR.predict(xtest.T)

ytrue = ytest.T



from sklearn.metrics import confusion_matrix

CM = confusion_matrix(ytrue,yprediciton)



#CM visualization



import seaborn as sns

import matplotlib.pyplot as plt



f, ax = plt.subplots(figsize=(5,5))

sns.heatmap(CM,annot = True, linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("Prediction(Ypred)")

plt.ylabel("Ytrue")

plt.show()