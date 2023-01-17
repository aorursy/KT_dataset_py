# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        

# Any results you write to the current directory are saved as output.


train_x = pd.read_csv("../input/titanic/train.csv", usecols = ['Age','Pclass', 'Sex','Fare', 
                                                               'Embarked', 'SibSp','Parch']) 
train_y = pd.read_csv("../input/titanic/train.csv", usecols = ['Survived']) 
test_x = pd.read_csv("../input/titanic/test.csv", usecols = ['Age','Pclass', 'Sex', 'Fare',  'Embarked',
                                                            'SibSp','Parch'])
test_x.head()
#train_y.isnull().sum()
train_x.isnull().sum()
#test_x.isnull().sum()
from statistics import mode

train_x.loc[train_x.Age.isnull(), 'Age'] = train_x.groupby("Pclass").Age.transform('median')
test_x.loc[test_x.Age.isnull(), 'Age'] = test_x.groupby("Pclass").Age.transform('median')
test_x.loc[test_x.Fare.isnull(), 'Fare'] = test_x.groupby("Pclass").Fare.transform('median')
train_x["Embarked"] = train_x["Embarked"].fillna(mode(train_x["Embarked"]))


test_x.tail()

train_x.isnull().sum()

sns.heatmap(train_x.corr(), annot = True)
'''for i in range(len(train_x)):
    if train_x['Sex'][i] == 'male':
        train_x['Sex'][i] = 0
    else:
        train_x['Sex'][i] = 1
    
#print(len(train_x))
#print(train_x['Sex'][0])
train_x.tail()'''

train_x["Sex"][train_x["Sex"] == "male"] = 0
train_x["Sex"][train_x["Sex"] == "female"] = 1
test_x["Sex"][test_x["Sex"] == "male"] = 0
test_x["Sex"][test_x["Sex"] == "female"] = 1

train_x["Embarked"][train_x["Embarked"] == "S"] = 0
train_x["Embarked"][train_x["Embarked"] == "C"] = 1
train_x["Embarked"][train_x["Embarked"] == "Q"] = 2

test_x["Embarked"][test_x["Embarked"] == "S"] = 0
test_x["Embarked"][test_x["Embarked"] == "C"] = 1
test_x["Embarked"][test_x["Embarked"] == "Q"] = 2



train_x_vec = np.array(train_x, dtype=np.float)
train_y_vec = np.array(train_y, dtype=np.float)
test_x_vec = np.array(test_x, dtype=np.float)



train_x_vec = train_x_vec.transpose()
train_y_vec = train_y_vec.transpose()
test_x_vec = test_x_vec.transpose()

print(test_x_vec)

def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

   
    s = 1 / (1 + np.exp(-z))

    
    return s
def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    
    w = np.zeros([dim, 1])
    b = 0
    

    #assert(w.shape == (dim, 1))
    #assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b
#print(w.T,b)
#print(np.dot(w.T, train_x_vec))

def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """
    
    m = X.shape[1]
    
    # FORWARD PROPAGATION (FROM X TO COST)

    A = sigmoid(np.dot(w.T, X) + b)          # compute activation
    #print(A)
    cost = -1/m* (np.dot(Y, np.log(A).T) + np.dot(1-Y, np.log(1-A).T) )    # compute cost
    #print(cost)

    #print(A.shape)
    #print(Y.shape)
    #print(1-Y)
    
    
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
  
    dw = 1/m * np.dot(X, (A-Y).T)
    

    db = 1/m * (np.sum(A-Y))

    #print(dw.shape, w.shape)
    #print(w, dw, db)


    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost
w, b = initialize_with_zeros(len(train_x_vec))
print(propagate(w, b, train_x_vec, train_y_vec))

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    
    costs = []
    
    for i in range(num_iterations):
        
        
        # Cost and gradient calculation (≈ 1-4 lines of code)
        grads, cost = propagate(w, b, X, Y)
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule (≈ 2 lines of code)
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

print(w,b, '!!!!!!!!!')


opt = optimize(w, b, train_x_vec, train_y_vec, 30000,  0.0005, print_cost = False)

w,b = opt[0]['w'], opt[0]['b']
print(w,b, opt[2][-1])



def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    #print(Y_prediction[0][0])
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T, X) + b)   
    #print(A)
    #print(A[0,0])


    for i in range(A.shape[1]):
        
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if A[0,i] > 0.5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0
       
        pass
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction
predic_train = predict(w, b, train_x_vec)
predict_test = predict(w, b, test_x_vec)
print("train accuracy: {} %".format(100 - np.mean(np.abs(predic_train - train_y_vec)) * 100))



final_prediction = pd.DataFrame(data=predict_test.T,index = [i for i in range(len(predict_test[0]))], columns=["Survived"])

final_prediction =  final_prediction["Survived"].astype(int)
#print(final_prediction)

test_x_Pid = pd.read_csv("../input/titanic/test.csv", usecols = ['PassengerId'])
submission = pd.concat([test_x_Pid, final_prediction], axis=1)
print(submission)

submission[['PassengerId', 'Survived']].to_csv('rf_submission_v1.csv', index = False)
