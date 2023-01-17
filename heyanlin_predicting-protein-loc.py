import numpy as np

import math

import pandas

import matplotlib.pyplot as plt

# load dataset

dataframe = pandas.read_csv("../input/ecoli.csv", delim_whitespace=True)

dataset = dataframe.values

print (dataset[0])

#print DataX.shape , DataY.shape 

DataX = np.array(dataset[:,0:8])

print (DataX.shape)

DataY = np.transpose([dataset[:,8]])

print (DataY.shape)



#print a samlpe

print ("input is : %s output is：%s" %(DataX[0], DataY[0]))

# Assign names to Columns

dataframe.columns = ['seq_name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'site']



dataframe = dataframe.drop('seq_name', axis=1)



# Encode Data

dataframe.site.replace(('cp', 'im', 'pp', 'imU', 'om', 'omL', 'imL', 'imS'),(1,2,3,4,5,6,7,8), inplace=True)
plt.hist((dataframe.site))
Training_Data = []

Test_Data = []

#This is a dataset split function(T : D = 8 : 2). The sample order is random，so we just follow the order selected 8/10 sample as Training_data, and the other as Test_data.

def split_sample():

    for i in range(math.ceil(len(dataset) * 0.8)):

        Training_Data.append([dataset[i]])

    print ("The Training_Data is : %s \n"% Training_Data[0])

    

    for i in range(math.ceil(len(dataset) * 0.8), len(dataset)):

        Test_Data.append(dataset[i])

    

    print ("The Test_Data is: %s"% Test_Data[0])

    

split_sample()
#Logistic Regression 

import numpy as np

import math 

debug = False

def debug(*argvs, **keywords):

    if debug:

        print (*argvs, **keywords)

#digmoid function 

def sigmoid(theta, DataX):

    Result = 1.0 / (1 + np.exp(-theta * DataX.T))

    return Result

#calculate the gredient

def gredient(theta, DataX, DataY):

    return (sigmoid(theta, DataX) - DataY)

#alpha is learning rate and iter_num is number of iterations

def gradientdescent(DataX, DataY, alpha = 0.001,iter_num = 2000):

    m, n = DataX.shape

    theta = np.mat(np.zero(n))

    for i in range(iter_num):

        theta -= alpha * gredient(theta, DataX, DataY)

    return theta

    
import numpy as np

import math

import pandas

import matplotlib.pyplot as plt

from sklearn.model_selection  import train_test_split

# load dataset

dataframe = pandas.read_csv("../input/ecoli.csv", delim_whitespace=True)

# Assign names to Columns

dataframe.columns = ['seq_name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'site']

dataframe = dataframe.drop('seq_name', axis=1)

# Encode Data

dataframe.site.replace(('cp', 'im', 'pp', 'imU', 'om', 'omL', 'imL', 'imS'),(1,0,0,0,0,0,0,0), inplace=True)

dataset = dataframe.values

#print (dataset[0])

#print dataset shape

DataX = np.array(dataset[:,0:7])

print (DataX.shape)

DataY = np.transpose([dataset[:,7]])

print (DataY.shape)
plt.hist((dataframe.site))
#dataset split

X_train,  X_test,Y_train, Y_test = train_test_split(DataX, DataY, test_size = 0.2)

print (X_train.shape, Y_train.shape)
#this is linear calculate part 

# theta is a vector of b,w  

#X is a sample feature matrix,each row represents a sample 

#theta.shape, result.shape ((7, 1), (268, 1))

def linear_cal(theta, X):

    result = np.dot( X,theta)

    return result
#the sigmoid function in order to calculate the y hat

#X is a sample feature matrix,each row represents a sample  

#shape is (1, 268)

def sigmoid(theta, X):

    result = 1.0 / (1 + np.exp(-linear_cal(theta, X)))

    return result
# cost function 

def costfunction(theta, X, Y):

    #cost =  -(Y * np.log(sigmoid(theta, X)) -  np.log(1 - sigmoid(theta, X)) * (1- Y)) / Y.size

    yhat = sigmoid(theta, X)

    L1 = Y * np.log(yhat)

    L2 = np.log(1 - sigmoid(theta, X)) * (1- Y)

    loss =  L1 - L2 

    cost = -loss/ Y.size

    return Y.shape, np.log(yhat).shape,np.transpose(sigmoid(theta, X)).shape, (1-Y).shape, L1.shape, L2.shape, loss.shape, cost.shape

costfunction(theta, X_train, Y_train)
#calculate the gredient 

#theta is matrix of db,dw 

#Y is corresponding to the labeled samples in X 

def gredient(theta, X, Y):

    m, n = X.shape

    theta = np.mat(np.zeros(n))

    gredient = (sigmoid(theta, X) - Y) * X / Y.size

    return gredient

#estimation parameters of gredient descent method, update dw, db

#alpha means step length ,we define the alpha equal to 0.001 

#iter_num is number of iterations ,we define the iter_num equal to 2000

def gradient_descent(X, Y,alpha, iter_num):

    m, n = X.shape

    theta = np.mat(np.zeros(n))

    J = []

    for i in range(iter_num):

        theta = theta - alpha * gredient(theta, X, Y)

        #cost = costfunction(theta, X, Y)

        #J.append(cost)

    return theta
# debug coding 

gradient_descent(X_train, Y_train, 0.001, 2000)
plt.hist((dataframe.site))
dataframe.plot(kind='density', subplots=True, layout=(3,4), sharex=False, sharey=False)
import numpy as np

import math

import pandas

import matplotlib.pyplot as plt

from sklearn.model_selection  import train_test_split



# load dataset

dataframe = pandas.read_csv("../input/ecoli.csv", delim_whitespace=True)

# Assign names to Columns

dataframe.columns = ['seq_name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'site']

dataframe = dataframe.drop('seq_name', axis=1)



# Encode Data

dataframe.site.replace(('cp', 'im', 'pp', 'imU', 'om', 'omL', 'imL', 'imS'),(1,0,0,0,0,0,0,0), inplace=True)

dataset = dataframe.values

#dataset divide

DataX = np.array(dataset[:,0:7])

DataY = np.transpose([dataset[:,7]])

print (DataX.shape, DataY.shape)

#dataset split

X_train, X_test,Y_train, Y_test = train_test_split(DataX, DataY, test_size = 0.2)



#Initialization parameters

def thetafunc(X):

    m, n = X.shape

    theta = np.mat(np.zeros(n))

    return theta



#this is linear calculate part 

# theta is a vector of b,w  

#X is a sample feature matrix,each row represents a sample 

def linear_cal(theta, X):

    result = np.dot(theta, X.T)

    return result



#the sigmoid function in order to calculate the y hat

#X is a sample feature matrix,each row represents a sample  

#shape is (1, 268)

def sigmoid(theta, X):

    result = 1.0 / (1 + np.exp(-linear_cal(theta, X)))

    return result 



# cost function 

def costfunction(theta, X, Y):

    #cost =  -(Y * np.log(sigmoid(theta, X)) -  np.log(1 - sigmoid(theta, X)) * (1- Y)) / Y.size

    yhat = sigmoid(theta, X)

    loss =  Y * np.log(yhat) + (1- Y) * np.log(1 - yhat)  

    cost = -np.sum(loss) / Y.size

    return cost



#calculate the gredient 

#theta is matrix of db,dw 

#Y is corresponding to the labeled samples in X 

def gredient(theta, X, Y):

    #theta = thetafunc(X)

    yhat = sigmoid(theta, X)

    gredient = (yhat - Y.T) * X / Y.size

    return gredient



#estimation parameters of gredient descent method, update dw, db

#alpha means step length ,we define the alpha equal to 0.001 

#iter_num is number of iterations ,we define the iter_num equal to 2000

def gradient_descent(X, Y,alpha, iter_num):

    theta = thetafunc(X)

    J = {}

    g = gredient(theta, X_train , Y_train)

    print (g)

    for i in range(iter_num):

        theta = theta - alpha * gredient(theta, X, Y)

        cost = costfunction(theta, X, Y)

        if (i % 10 == 0):

            J[i] =cost

    plt.plot(J.keys(), J.values())

    #print (J)

    return theta.shape



# debug coding 

gradient_descent(X_train, Y_train, 0.001, 2000)





###Using the test dataset to calculate the logist regression model  accuracy.

grads_theta = [[ 0.09533582,0.07391791,  0.04552239,  0.03824627,  0.05699627,  0.11410448, 0.07921642]]



###start code here###

def logist_predict(X_test, Y_test, theta):

    yhat = sigmoid(theta, X_test)

    acc = 100 * np.mean((np.dot(Y_test, yhat) + np.dot(1 - Y_test, 1 - yhat)) / float(Y_test.size))

  #  acc = 0

#    for i in range (0, Y_test.size):

#        if (predictions[0][i] == Y_test[0][i]):

#            acc += 1

#        else:

#            pass

                           

 #   return acc

###end code here###  

    return acc

###Print the accuracy

acc = logist_predict(X_test, Y_test, grads_theta)

print (acc)