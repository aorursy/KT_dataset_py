import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import h5py

import scipy

from PIL import Image
df = pd.read_csv("../input/insurance/insurance.csv")
df.head()
df.info()
categorical = [var for var in df.columns if df[var].dtype=='O']

print("The categorical features are : ",categorical)
df = pd.concat(      [df,

                     pd.get_dummies(df.sex), 

                     pd.get_dummies(df.smoker),

                     pd.get_dummies(df.region)], axis=1)
df.head()
df.drop(['sex'], axis=1, inplace=True)

df.drop(['smoker'], axis=1, inplace=True)

df.drop(['region'], axis=1, inplace=True)
df.head()
y = df["charges"]

df.drop(['charges'], axis=1, inplace=True)
df.isna().any()
df.isnull().any()
from sklearn.model_selection import train_test_split



train_set_x,  test_set_x, train_set_y,test_set_y = train_test_split(df, y, test_size = 0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler  

scaler = StandardScaler()  

scaler.fit(train_set_x)  

train_set_x = scaler.transform(train_set_x)  

test_set_x = scaler.transform(test_set_x)
print(train_set_y)
print("train_set_x",train_set_x.shape)

print("train_set_y",train_set_y.shape)

print("test_set_x",test_set_x.shape)

print("test_set_y",test_set_y.shape)

train_set_y=train_set_y.values.reshape(train_set_y.shape[0],1)

test_set_y=test_set_y.values.reshape(test_set_y.shape[0],1)

print("train_set_x",train_set_x.shape)

print("train_set_y",train_set_y.shape)

print("test_set_x",test_set_x.shape)

print("test_set_y",test_set_y.shape)
def initialize(dim):

    w = np.zeros((dim,1))

    b = 0  

    return w, b
def propagate(w, b, X, Y):

    m = X.shape[1]

    # FORWARD 

    A = np.dot(w.T,X)+b 

    cost = (1/(2*m))*np.sum((A - Y) ** 2)

    # BACKWARD 

    dz = A-Y

    dw = (1/m)*np.dot(X,dz.T)

    db = (1/m)*np.sum(dz)

    

    grads = {"dw": dw,

             "db": db}



    return grads, cost
def optimize(w, b, X, Y,print_cost, num_iterations, learning_rate):

    costs = []

    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, Y)

        

        dw = grads["dw"]

        db = grads["db"]



        w = w-learning_rate*dw

        b = b-learning_rate*db



        if i % 100 == 0:

            costs.append(cost)

        

        if print_cost and i % 100 == 0:

            print ("Cost after iteration %i: %f" %(i, cost))

    

    params = {"w": w,

              "b": b}

    grads = {"dw": dw,

             "db": db}

    

    return params, grads, costs
def predict(w, b, X):

    m = X.shape[1]

    Y_prediction = np.zeros((1,m))

    w = w.reshape(X.shape[0], 1)

    

    Y_prediction = np.dot(w.T,X)+b

    

    return Y_prediction 
from sklearn.metrics import r2_score

def model(X_train, Y_train, X_test, Y_test,print_cost, num_iterations = 2000, learning_rate = 0.5 ):

    w, b = initialize(X_train.shape[0])



    parameters, grads, costs = optimize(w, b, X_train, Y_train,print_cost, num_iterations, learning_rate)



    w = parameters["w"]

    b = parameters["b"]



    Y_prediction_test = predict(w, b, X_test)

    Y_prediction_train = predict(w, b, X_train)

    

    print("train accuracy: {} %".format(r2_score(Y_train.T, Y_prediction_train.T)))

    print("test accuracy: {} %".format(r2_score(Y_test.T, Y_prediction_test.T)))



    

    d = {"costs": costs,

         "Y_prediction_test": Y_prediction_test, 

         "Y_prediction_train" : Y_prediction_train, 

         "w" : w, 

         "b" : b,

         "learning_rate" : learning_rate,

         "num_iterations": num_iterations}

    

    return d
d = model(train_set_x.T, train_set_y.T, test_set_x.T, test_set_y.T,print_cost = True, num_iterations = 1000, learning_rate = 0.01)
learning_rates = [0.1, 0.01, 0.001, 0.0001]

models = {}

for i in learning_rates:

    print ("learning rate is: " + str(i))

    models[str(i)] = model(train_set_x.T, train_set_y.T, test_set_x.T, test_set_y.T,print_cost = False, num_iterations = 2000, learning_rate = i)

    print ('\n' + "-------------------------------------------------------" + '\n')



for i in learning_rates:

    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))



plt.ylabel('cost')

plt.xlabel('iterations (hundreds)')



legend = plt.legend(loc='upper center', shadow=True)

frame = legend.get_frame()

frame.set_facecolor('0.90')

plt.show()
from sklearn.linear_model import LinearRegression



lr = LinearRegression().fit(train_set_x,train_set_y)

y_train_pred = lr.predict(train_set_x)

y_test_pred = lr.predict(test_set_x)

print(lr.score(train_set_x,train_set_y))

print(lr.score(test_set_x,test_set_y))
from sklearn.linear_model import Ridge

reg = Ridge(alpha=.5).fit(train_set_x,train_set_y)

y_train_pred = lr.predict(train_set_x)

y_test_pred = lr.predict(test_set_x)

print(reg.score(train_set_x,train_set_y))

print(reg.score(test_set_x,test_set_y))