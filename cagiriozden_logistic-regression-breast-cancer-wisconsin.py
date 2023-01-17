import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



df = pd.read_csv("../input/data.csv")

print(df.info())

#remove unused rows from data frame

df.drop(["id","Unnamed: 32"],axis=1,inplace=True)
B = df[df.diagnosis=="B"]

M = df[df.diagnosis=="M"]

print("Benign counts : ", B.shape)

print("Malignant counts : ",M.shape)



plt.scatter(B.area_mean ,B.texture_mean,color="green")

plt.scatter(M.area_mean ,M.texture_mean,color="red")

plt.show()



#Update diagnosis results with 1 and 0.Benign = 0 , Malignant= 1

df.diagnosis = [1 if each == "M" else 0 for each in df.diagnosis]



y = df.iloc[:,0].values.reshape(-1,1)

X = df.drop("diagnosis",axis=1)



#Normalization..

X = (X - np.min(X)) / (np.max(X) - np.min(X)).values
def sigmoid(z) :

    y_head = 1/(1+np.exp(-z))

    return y_head
def initWeightsAndBias(size):

    w = np.full((size,1),0.01)

    b = 0.0

    return w, b
# z = w.T*x+b

def forwardPropogation (w,b,x_train,y_train,y_head):

    v1 = -y_train*np.log(y_head)

    v2 = (1-y_train)*np.log(1-y_head)

    loss = v1-v2

    cost = (np.sum(loss))/x_train.shape[1]      # divide to size of x_train for scaling

    return cost
def backwardPropogation (x_train,y_train,y_head):

    calculated_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]

    calculated_bias = np.sum(y_head-y_train)/x_train.shape[1]

    gradients = {"calculated_weight": calculated_weight,"calculated_bias": calculated_bias}

    return gradients
def forwardBackwardPropagation(w,b,x_train,y_train):

    z = np.dot(w.T,x_train) + b

    y_head = sigmoid(z)

    cost = forwardPropogation(w,b,x_train,y_train,y_head)

    gradients = backwardPropogation(x_train,y_train,y_head)

    return cost,gradients
def learn(w, b, x_train, y_train, learningRate,iteration):

    costList = []

    

    for i in range(iteration):

        # execute forward backward propogation to get updated gradients and cost

        cost,gradients = forwardBackwardPropagation(w,b,x_train,y_train)

        costList.append(cost)

        # update weight and bias with the calculated values

        w = w - learningRate * gradients["calculated_weight"]

        b = b - learningRate * gradients["calculated_bias"]

        #if (i % 100 == 0) :

        #    print("w : {} - b : {} ".format(w,b) )

        

    # updated weights and bias

    index = np.arange(iteration)

    parameters = {"weight": w,"bias": b}

    #print("Cost lists : " , costList)

    plt.plot(index,costList)

    plt.xticks(index,rotation=90)

    plt.xlabel(" - Iteration Count - ")

    plt.ylabel(" - Cost - ")

    plt.show()

    return parameters, gradients, costList
def predict(w,b,x_test):

    # make some prediction with test data..

    z = sigmoid(np.dot(w.T,x_test)+b)

    y_prediction = np.zeros((1,x_test.shape[1]))

    

    # Limit is 0.5. If predicted value is greater than 0.5 

    # then i'll flag the tumor as a malignant otherwise i'll flag it as benign 

    # We can change limit value if its necessary..

    for i in range(z.shape[1]):

        if z[0,i]> 0.5:

            y_prediction[0,i] = 1

        



    return y_prediction
def logisticRegression(x_train, y_train, x_test, y_test, learningRate ,  iteration):

    # initialize

    size =  x_train.shape[0]

    w,b = initWeightsAndBias(size)

    parameters, grad, costList = learn(w, b, x_train, y_train, learningRate,iteration)

    

    y_test_predicted = predict(parameters["weight"],parameters["bias"],x_test)



    # Calculate accuracy with test set..

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_test_predicted - y_test)) * 100))
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

x_train = x_train.T

x_test = x_test.T

y_train = y_train.T

y_test = y_test.T



# print("x_train shape : " ,x_train.shape)

# print("x_test shape : " ,x_test.shape)

# print("y_train shape : " ,y_train.shape)

# print("y_test shape : " ,y_test.shape)



#Execute logistic regression for training..

logisticRegression(x_train, y_train, x_test, y_test,learningRate=1,iteration=150)