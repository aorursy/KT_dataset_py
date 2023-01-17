""" 

    Code : Simple Linear Regression 

    Created By : Gitanjali Murumkar

    

"""



import pandas as pd

import numpy as np



def importing_dataset():

    DataSet = pd.read_csv('../input/random-salary-data-of-employes-age-wise/Salary_Data.csv')

    X =DataSet.iloc[ : , 0].values

    Y =DataSet.iloc[ : ,  -1].values

    return X,Y

   

def train_test_data (X,Y):

    from sklearn.model_selection import train_test_split

    X_train,X_test,Y_train,Y_test = train_test_split (X,Y,test_size=0.2, random_state=0)

    return X_train,X_test,Y_train,Y_test

    

def train_mean(X_train,Y_train):

    X_train_mean = X_train.mean()

    Y_train_mean = Y_train.mean()

    return X_train_mean,Y_train_mean



def mean_of_each_element_sq(X_train,Y_train):   

    sum_x_sq=0

    for x in np.nditer(X_train):

          sum_x_sq =sum_x_sq + x*x 

    X_train_sq_mean =sum_x_sq /len(X_train)        

    sum_y_sq=0

    for y in np.nditer(Y_train):

        sum_y_sq =sum_y_sq + y*y

    Y_train_sq_mean =sum_y_sq /len(X_train)       

    return sum_x_sq,sum_y_sq,X_train_sq_mean, Y_train_sq_mean



def mul_xtrain_ytrain(X_train,Y_train):

    sum_train = X_train.sum()*Y_train.sum()

    return (sum_train)   



def mul_sum_xtrain_ytrain(X_train,Y_train):

    mul_sum_train = np.multiply(X_train , Y_train).sum()

    return (mul_sum_train) 



def sum_xtrain_y_train(X_train,Y_train):

    X_train_sum =np.sum(X_train)

    Y_train_sum =np.sum(Y_train)

    return(X_train_sum,Y_train_sum) 

    

    

#importing data

X,Y = importing_dataset()

#divide data into train and test data set

X_train,X_test,Y_train,Y_test = train_test_data (X,Y)

#find mean 

X_train_mean,Y_train_mean = train_mean(X_train,Y_train)

#find square of each element

sum_x_sq,sum_y_sq,X_train_sq_mean, Y_train_sq_mean= mean_of_each_element_sq(X_train,Y_train)



X_train_sum,Y_train_sum = sum_xtrain_y_train(X_train,Y_train)



Sxx = sum_x_sq-((X_train_sum*X_train_sum)/len(X_train) )

Syy = sum_y_sq-((Y_train_sum*Y_train_sum)/len(Y_train) )

Sxy = mul_sum_xtrain_ytrain(X_train,Y_train) - mul_xtrain_ytrain(X_train,Y_train)/len(X_train)

B1 = Sxy/Sxx

B0= Y_train_mean - (B1*X_train_mean)



# predict y test 

Y_test_predict=[]

for i in range(len(X_test)):

    Y_test_predict.append(B0 + B1*X_test[i]) 

print("Y_test_predict",Y_test_predict)

print("Y_test",Y_test)



SST=[]

SSR=[]

for n in range(len(X_test)):

    SST.append((Y_test[n] - Y_test.mean())*(Y_test[n] - Y_test.mean()))

    SSR.append((Y_test_predict[n]-Y_test.mean())*(Y_test_predict[n]-Y_test.mean()))



R_sq = np.mean(SSR)/np.mean(SST)

print ("R_Square",R_sq)