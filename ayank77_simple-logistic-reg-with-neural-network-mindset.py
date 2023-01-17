import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv("../input/train.csv")

df.shape
df.columns
def only1(x):

    if x == 1:

        return x

    else:

        return 0

df["label"]=df.label.apply(only1)
df.label.value_counts()
df1 = df[df.label == 1]

df2 = df[df.label == 0]

df2 = df2.sample(n=4684)

frame = [df1,df2]

df_final = pd.concat(frame)

df_final.shape

df_final = df_final.sample(frac=1).reset_index(drop=True)

df_final.head(5)
df_y = df_final["label"]

df_x = df_final.iloc[: , 1:]

df_y.values.reshape(1,9368)
from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.3, random_state=42)
print (x_train.shape)

print (x_test.shape)

print (y_train.shape)

print (y_test.shape)
x_test = x_test.T

x_train = x_train.T

y_train = y_train.reshape(1,6557)

y_train = y_train.T

y_test = y_test.reshape(1,2811)

y_test = y_test.T
y_train = y_train.T
y_test = y_test.T
x_train = x_train/255

x_test = x_test/255
def sigmoid(x):

    sig = 1/(1 + np.exp(-x))

    return sig 
def initialize_with_zeroes(dim):

    w = np.zeros((dim,1),dtype=np.float)

    b = np.zeros((1,1), dtype=np.float)

    return w,b
def propagate(w,b, x, y):

    m = x.shape[1]

    A = sigmoid(np.dot(w.T,x) + b)

    cost =np.sum(np.dot(y,np.log(A).T)+ np.dot((1-y),np.log(1-A).T))/(-m)

    dw = (np.dot(x,(A-y).T))/(m)

    db = (np.sum(A-y))/(m)

    cost = np.squeeze(cost)

    grads = {"dw": dw,

             "db": db}

    return grads, cost

    
def optimize(w, b, x, y, num_iterations, learning_rate, print_cost = False):

    costs = []

    for i in range(num_iterations):

        grads, cost = propagate(w, b, x,y)

        dw = grads["dw"]

        db = grads["db"]

        w = w-np.dot(learning_rate,dw)

        b = b-np.dot(learning_rate,db)

        if i % 100 == 0:

            costs.append(cost)

        if print_cost and i % 100 == 0:

            print ("Cost after iteration %i: %f" %(i, cost))

        params = {"w": w,

                  "b": b}

        grads = {"dw": dw,

                 "db": db}

        return params, grads, costs
def predict(w, b, x):

    m = x.shape[1]

    Y_prediction = np.zeros((1,m))

    w = w.reshape(x.shape[0], 1)

    A = sigmoid(np.dot(w.T,x)+b)

    for i in range(A.shape[1]):

        if A[0,i]<=0.5:

            Y_prediction[0][i]=0

        elif A[0,i]>0.5:

            Y_prediction[0][i]=1

    return Y_prediction
def model(X_train, Y_train, X_test, Y_test, num_iterations = 5000, learning_rate = 0.5, print_cost = False):

    w, b = initialize_with_zeroes(X_train.shape[0])

    parameters, grads, costs = optimize(w, b, X_train, Y_train , num_iterations, learning_rate, print_cost = False)

    # Retrieve parameters w and b from dictionary "parameters"

    w = parameters["w"]

    b = parameters["b"]

    # Predict test/train set examples (≈ 2 lines of code)

    Y_prediction_test =predict(w,b,X_test)

    Y_prediction_train =predict(w, b, X_train) 



    ### END CODE HERE ###



    # Print train/test Errors

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))

    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))



    

    d = {"costs": costs,

         "Y_prediction_test": Y_prediction_test, 

         "Y_prediction_train" : Y_prediction_train, 

         "w" : w, 

         "b" : b,

         "learning_rate" : learning_rate,

         "num_iterations": num_iterations}

    

    return d
d = model(x_train, y_train, x_test, y_test , num_iterations = 1000, learning_rate = 0.005, print_cost = True)