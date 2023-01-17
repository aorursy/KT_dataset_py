# For data creation and other tasks
import numpy as np
import random
import scipy
from scipy.stats import norm
import pandas as pd

## Mertics to evaluate the models 
from sklearn.metrics import accuracy_score # for Logistic Regression

# For plotting the graphs.. 
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns 

# For spliting the data into 80:20 ratio 
from sklearn.model_selection import train_test_split
def generate_data(n,m,theta):
    X = []
    # m = 3  #change to user defined 
    # n = 100   #change to user defined 
    theta = int(n*(theta))  #change to user defined 
    for i in range(0,n):
        X_i = scipy.stats.norm.rvs(0,1,m)
        X.append(X_i)
    
    beta = scipy.stats.norm.rvs(0,1,m)
    # for simplicity I am not adding '1' to either beta or X rather directly adding it to the 'odds' that will
    ## be used as y1 which will be passed through cost fn with theta which will define whether it will be 
    ### '1' or '0' (bernolli distribution)
    odds =  (np.exp(1+ np.matmul(X,beta)) / ( 1 + np.exp(1+ np.matmul(X,beta)) )) 
    y1 = []
    for i in odds:
        if(i >= 0.5):
            y1.append(1)
        else:
            y1.append(0)
    df1 = pd.DataFrame(X)
    df2 = pd.DataFrame(y1)
    df1['Y'] = df2[0]
    #df1.head()
    #df1.tail()

    ## Adding noise using theta 
    change = df1.sample(theta).index
    
    for i in change:
        if(df1.loc[i,'Y'] == 0):
            df1.loc[i,'Y'] = 1 
        else:
            df1.loc[i,'Y'] = 0 
    return df1
def weightInitialization(n_features):
    w = np.zeros((1,n_features))
    b = 0
    return w,b
def sigmoid_activation(result):
    final_result = 1/(1+np.exp(-result))
    return final_result

def model_optimize(w, b, X, Y):
    m = X.shape[0]
    
    #Prediction
    final_result = sigmoid_activation(np.dot(w,X.T)+b)
    Y_T = Y.T
    cost = (-1/m)*(np.sum((Y_T*np.log(final_result)) + ((1-Y_T)*(np.log(1-final_result)))))
    #
    
    #Gradient calculation
    dw = (1/m)*(np.dot(X.T, (final_result-Y.T).T))
    db = (1/m)*(np.sum(final_result-Y.T))
    
    grads = {"dw": dw, "db": db}
    
    return grads, cost
def model_predict(w, b, X, Y, learning_rate, no_iterations):
    costs = []
    for i in range(no_iterations):
        #
        grads, cost = model_optimize(w,b,X,Y)
        #
        dw = grads["dw"]
        db = grads["db"]
        #weight update
        w = w - (learning_rate * (dw.T))
        b = b - (learning_rate * db)
        #
        
        if (i % 100 == 0):
            costs.append(cost)
            #print("Cost after %i iteration is %f" %(i, cost))
    
    #final parameters
    coeff = {"w": w, "b": b}
    gradient = {"dw": dw, "db": db}
    
    return coeff, gradient, costs
def predict(final_pred, m):
    y_pred = np.zeros((1,m))
    for i in range(final_pred.shape[1]):
        if final_pred[0][i] > 0.5:
            y_pred[0][i] = 1
    return y_pred

def Log_Res(n,m,theta,learning_rate,no_iterations):
    df1 = generate_data(n,m,theta)
        
    X1 = df1.iloc[:,0:m].values
    y1 = df1.iloc[:,m].values
        
    X_train,X_test,Y_train,Y_test = train_test_split(X1,y1,test_size = 0.2)
    n_features = X_train.shape[1]
    print('Number of Features', n_features)
    w, b = weightInitialization(n_features)
    #Gradient Descent
    coeff, gradient, costs = model_predict(w, b, X_train, Y_train,learning_rate=0.0001,no_iterations=45000)
    #Final prediction
    w = coeff["w"]
    b = coeff["b"]
    print('Optimized weights - Beta', w)
    print('Optimized intercept',b)
    #
    final_train_pred = sigmoid_activation(np.dot(w,X_train.T)+b)
    final_test_pred = sigmoid_activation(np.dot(w,X_test.T)+b)
    #
    m_tr =  X_train.shape[0]
    m_ts =  X_test.shape[0]
       #
    y_tr_pred = predict(final_train_pred, m_tr)
    print('Training Accuracy',accuracy_score(y_tr_pred.T, Y_train))
    #
    y_ts_pred = predict(final_test_pred, m_ts)
    print('Test Accuracy',accuracy_score(y_ts_pred.T, Y_test))
    
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title('Cost reduction over time')
    plt.show()
## when there is no noise 
Log_Res(200,3,0.01,learning_rate=0.0001,no_iterations=5000)
## when there is some noise (10% noise)
Log_Res(200,3,0.1,learning_rate=0.0001,no_iterations=5000)
## when there is a lot of noise (30% noise)
Log_Res(200,3,0.3,learning_rate=0.0001,no_iterations=5000)
## when n is small n = 50
Log_Res(50,3,0.0,learning_rate=0.0001,no_iterations=5000)
## when n is large n = 500
Log_Res(500,3,0.0,learning_rate=0.0001,no_iterations=5000)
## when n is very large n = 1000
Log_Res(10000,3,0.0,learning_rate=0.0001,no_iterations=5000)
