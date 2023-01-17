# Importing basic libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.utils import shuffle
# Import Dataset

dataset = pd.read_csv("../input/kc_house_data.csv")
dataset.head()
dataset.describe()



#taking a close look at dataset to analyze it.
# Checking missing data

dataset.isnull().sum()
# Dropping unnecessary columns

dataset.drop(['id', 'date', 'waterfront', 'view', 'zipcode', 'long', 'sqft_basement', 'yr_renovated'], axis=1, inplace = True)
dataset.head()
dataset['grade'].value_counts()
# Normalize dataset



dataset =(dataset-dataset.mean())/dataset.std()
x = dataset.iloc[:, 1:].values

y = dataset.iloc[:, :1].values
# Splitting dataset into training and testing part



from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size = 0.3)
N, D = x.shape                # Dimentions of data

K = 10                        # No. of hidden units
# Initializing random weights and bias for our neural network



w1 = np.random.randn(D,K)/np.sqrt(D+K)

b1 = np.zeros(K)

w2 = np.random.randn(K,1)/np.sqrt(K+1)

b2 = 0
# Feed forward function for ANN

def feed_forward(x, w1, b1, w2, b2):

    zout = np.maximum((x.dot(w1)+b1), 0)          # relu nonlinearity

    return (zout.dot(w2)+b2), zout                # we'll return both of them.
# Differentiation of relu will be required.

d_relu = lambda x: (x>0).astype(x.dtype)
# Describing Cost function for regression.

def cost(T, Y):

    return  ((T-Y)*(T-Y)).sum()/N

# Empty list, cost will be appended in backpropagation 

train_cost = []

test_cost = []



lr =  0.01                            # Learning Rate

reg = 0                               # Regularization 
# For batch gradient descent

iter = 1000                                         # No. of iteration

batch_size = 300                                    # Batch Size

n_batches = np.round(len(xtrain)/batch_size)        # No. of batches

n_batches = n_batches.astype(int)
# I am using batch gradient descent, so I'll comment code for full gradient descent



# FULL GRADIENT DESCENT

#for i in range(10000):

#    ypred, ztrain = feed_forward(xtrain, w1, b1, w2, b2)

#    trainCost = cost(ytrain, ypred)

    

#    ytest_pred, ztest = feed_forward(xtest, w1, b1, w2, b2)

#    testCost = cost(ytest, ytest_pred)

    

#    if i%1000==0:

#        print(trainCost)

    

#    E = ypred-ytrain

#    w2-= lr*(ztrain.T.dot(E)/len(xtrain) + reg*w2)

#    b2-= lr*(E.sum()/len(xtrain) + reg*b2)

    

#    dz = E.dot(w2.T)*d_relu(ztrain)

#    w1-= lr*(xtrain.T.dot(dz)/ len(xtrain) + reg*w1)

#    b1-= lr*(dz.sum()/len(xtrain) + reg*b1)

    

#    train_cost.append(trainCost)

#    test_cost.append(testCost)

    



# BATCH GRADIENT DESCENT

for i in range(iter):

    tempX, tempY = shuffle(xtrain, ytrain)



    for j in range(n_batches):

        batchX = tempX[j*batch_size:(j*batch_size + batch_size), :]

        batchY = tempY[j*batch_size:(j*batch_size + batch_size), :]

        ypred, ztrain = feed_forward(batchX, w1, b1, w2, b2)

        

    

        trainCost = cost(batchY, ypred)

    

        ytest_pred, ztest = feed_forward(xtest, w1, b1, w2, b2)

        testCost = cost(ytest, ytest_pred)

    

        if i%50 ==0 and j%50==0:

            print(testCost)

    

        E = ypred-batchY

        w2-= lr*(ztrain.T.dot(E) + reg*w2)/len(batchX)

        b2-= lr*(E.sum() + reg*b2)/len(batchX)

    

        dz = E.dot(w2.T)*d_relu(ztrain)

        w1-= lr*(batchX.T.dot(dz) + reg*w1)/ len(batchX)

        b1-= lr*(dz.sum() + reg*b1)/len(batchX)

    

        train_cost.append(trainCost)

        test_cost.append(testCost)
# Plotting train and test cost.



plt.plot(train_cost, 'k') #black

plt.plot(test_cost, 'b') #blue
ytest_pred, ztest = feed_forward(xtest, w1, b1, w2, b2)         # Our prediction



d1 = ytest-ytest_pred

d2 = ytest-ytest.mean()

r2 = 1-d1.T.dot(d1)/d2.T.dot(d2)                                # R-Squared value



print(r2)