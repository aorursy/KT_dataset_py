# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn.utils # Shuffle Pandas data frame



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#2: Read data

data_path = "/kaggle/input/learn-together/train.csv"

data = pd.read_csv(data_path)

data = sklearn.utils.shuffle(data)

print('data readed successfully :)')
#3: Data Explor



#first 5 rows in data

print(data.head())



#describe data

print(data.describe())



#data dimintionality [rows x columns]

print(data.shape)

#number of classes in target column

num_of_classes = len(data['Cover_Type'].unique())

print(num_of_classes)
#4 Prepare data for training and cross-validation purpose



#drop Id column

data.drop('Id', axis=1, inplace=True)



#add ones column

data.insert(0,'ones',1)

train_data = data



#split train data to featuers and target 

X_train_data = train_data.iloc[ : , 0:train_data.shape[1]-1]

y_train_data = train_data.iloc[ : , train_data.shape[1]-1:]



print('data preparepared successfully :)')
#5 Hyper parameters

thetas = np.zeros((num_of_classes,X_train_data.shape[1]))

alpha = 1e-22
#6 Sigmoid function

def sigmoid(z):

    return 1 / (1 + np.exp(-z))



print('Sigmoid function created :)')
#7 Cost Function

def computeCost(theta,X, y,alpha):

    theta = np.matrix(theta)

    X = np.matrix(X)

    y = np.matrix(y)



    firstTerm  = np.multiply(y, np.log(sigmoid(X * theta.T)))

    secondTerm = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))

    reg        = (alpha / 2*len(X)) * np.sum(np.power(theta[:,1:theta.shape[1]],2))

    cost         = -np.sum(firstTerm + secondTerm) / len(X) + reg

    return cost



print('Cost Function created :)')
#8: Gradient function

def gradient(theta, X, y, alpha): 

    theta = np.matrix(theta)

    X  = np.matrix(X)

    y  = np.matrix(y)

   

    error = sigmoid(X * theta.T) - y

    #refere to gradient value

    grad = ((X.T*error) / len(X)).T + ((alpha / len(X)) * theta)

    grad[0,0] = np.sum(np.multiply(error,X[:,0])) / len(X)

        

    return np.array(grad).ravel()



print('Gradient function created :)')
#9 one_vs_all function

from scipy.optimize import minimize



def one_vs_all(X,y,number_of_classes,alpha):

    

    rows = X.shape[0]

    columns = X.shape[1]

    all_thetas = np.zeros((num_of_classes,columns))

    

    for i in range (1,num_of_classes+1):

        theta = np.zeros(columns)

        y_i = np.array([ 1 if target == i else 0 for target in y['Cover_Type'] ])

        y_i = np.reshape(y_i,(rows,1))

        

        func_min = minimize(fun = computeCost, x0 = theta , args = (X,y_i,alpha) , method='TNC' , jac=gradient)

        all_thetas[i-1,:] = func_min.x

        

    return all_thetas



all_thetas = one_vs_all(X_train_data,y_train_data,num_of_classes,alpha)

print(all_thetas)
#10 Predict all function

def predictAll(X,all_thetas):   

    X = np.matrix(X)

    all_thetas = np.matrix(all_thetas)

    result = sigmoid(X * all_thetas.T)

    res_argmax = np.argmax(result , axis = 1)

    res_argmax = res_argmax + 1

    return res_argmax



test_data_path = "/kaggle/input/learn-together/test.csv"

test_data = pd.read_csv(test_data_path,index_col='Id')



test_data_id = test_data.index

test_data.insert(0,'ones',1)



y_test_predict = predictAll(test_data,all_thetas)



y_test_predict = np.array(y_test_predict)

y_test_predict = y_test_predict.T

y_test_predict = y_test_predict.ravel() 



output = pd.DataFrame({

        'Id':test_data_id,

        'Cover_Type':y_test_predict

        })



output.to_csv('submission.csv', index=False)  

print('Go :)')