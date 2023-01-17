
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
def predict(final_pred, m):
    y_pred = np.zeros((1,m))
    for i in range(final_pred.shape[1]):
        if final_pred[0][i] > 0.5:
            y_pred[0][i] = 1
    return y_pred
lam = 0.1
def model_optimize_for_L1(w, b, X, Y):
    m = X.shape[0]
    
    #Prediction
    final_result = sigmoid_activation(np.dot(w,X.T)+b)
    Y_T = Y.T
    cost = (-1/m)*(np.sum((Y_T*np.log(final_result)) + ((1-Y_T)*(np.log(1-final_result)))))  + (lam * (np.sum(w)))

    #
    
    #Gradient calculation
    dw = (1/m)*(np.dot(X.T, (final_result-Y.T).T)) + lam
    db = (1/m)*(np.sum(final_result-Y.T))
    
    grads = {"dw": dw, "db": db}
    
    return grads, cost
def model_predict_l1(w, b, X, Y, learning_rate, no_iterations):
    costs = []
    for i in range(no_iterations):
        #
        grads, cost = model_optimize_for_L1(w,b,X,Y)
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
lam = 0.1
def model_optimize_for_L2(w, b, X, Y):
    m = X.shape[0]
    
    #Prediction
    final_result = sigmoid_activation(np.dot(w,X.T)+b)
    Y_T = Y.T
    cost = (-1/m)*(np.sum((Y_T*np.log(final_result)) + ((1-Y_T)*(np.log(1-final_result)))))  + (lam * (np.sum(np.square(w))))

    #
    
    #Gradient calculation
    dw = (1/m)*(np.dot(X.T, (final_result-Y.T).T)) + lam * w
    db = (1/m)*(np.sum(final_result-Y.T))
    
    grads = {"dw": dw, "db": db}
    
    return grads, cost
def model_predict_l2(w, b, X, Y, learning_rate, no_iterations):
    costs = []
    for i in range(no_iterations):
        #
        grads, cost = model_optimize_for_L2(w,b,X,Y)
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
def Log_reg_L1_L2(n,m,theta,learning_rate,no_iterations):
    df1 = generate_data(n,m,theta)
        
    X1 = df1.iloc[:,0:m].values
    y1 = df1.iloc[:,m].values
        
    X_train,X_test,Y_train,Y_test = train_test_split(X1,y1,test_size = 0.2)
    n_features = X_train.shape[1]
    print('Number of Features', n_features)
    w, b = weightInitialization(n_features)
    #Gradient Descent for L1 reguralisation
    coeff1, gradient1, costs1 = model_predict_l1(w, b, X_train, Y_train,learning_rate=0.0001,no_iterations=45000)
     
    #Final prediction
    w1 = coeff1["w"]
    b1 = coeff1["b"]
    print('Optimized weights - Beta', w1)
    print('Optimized intercept',b1)
    #
    final_train_pred = sigmoid_activation(np.dot(w1,X_train.T)+b1)
    final_test_pred = sigmoid_activation(np.dot(w1,X_test.T)+b1)
    #
    m_tr =  X_train.shape[0]
    m_ts =  X_test.shape[0]
       #
    y_tr_pred = predict(final_train_pred, m_tr)
    print('Training Accuracy',accuracy_score(y_tr_pred.T, Y_train))
    #
    y_ts_pred = predict(final_test_pred, m_ts)
    print('Test Accuracy',accuracy_score(y_ts_pred.T, Y_test))
    
    plt.plot(costs1)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title('Cost reduction over time using L1 Regularisation')
    plt.show()
    
    
    #Gradient Descent for L2 reguralisation
    coeff2, gradient2, costs2 = model_predict_l2(w, b, X_train, Y_train,learning_rate=0.0001,no_iterations=45000)
    #Final prediction
    w2 = coeff2["w"]
    b2 = coeff2["b"]
    print('Optimized weights - Beta', w2)
    print('Optimized intercept',b2)
    #
    final_train_pred = sigmoid_activation(np.dot(w2,X_train.T)+b2)
    final_test_pred = sigmoid_activation(np.dot(w2,X_test.T)+b2)
    #
    m_tr =  X_train.shape[0]
    m_ts =  X_test.shape[0]
       #
    y_tr_pred = predict(final_train_pred, m_tr)
    print('Training Accuracy',accuracy_score(y_tr_pred.T, Y_train))
    #
    y_ts_pred = predict(final_test_pred, m_ts)
    print('Test Accuracy',accuracy_score(y_ts_pred.T, Y_test))

    plt.plot(costs2)
    plt.ylabel('cost2')
    plt.xlabel('iterations (per hundreds)')
    plt.title('Cost reduction over time using L2 Regularisation')
    plt.show()
## when using L1 reg and L2 reg
Log_reg_L1_L2(200,3,0.01,learning_rate=0.0001,no_iterations=5000)
