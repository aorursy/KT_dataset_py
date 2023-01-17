# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

print(len(train_data),len(train_data.columns))

train_data.head()
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

print(len(test_data),len(test_data.columns))

test_data.head()
women = train_data.loc[train_data.Sex=='female']['Survived']

rate_women = sum(women)/len(women)

print('fraction of women who survived:',rate_women)
men = train_data.loc[train_data.Sex=='male']['Survived']

rate_men = sum(men)/len(men)

print('fraction of men who survived:',rate_men)
from sklearn.model_selection import train_test_split
def split_data(X_train,Y_train):

    '''

    Split the training data to create a train set and a cross validation set.

    

    Argument:

    X_train: Pandas DataFrame object of dimensions (889,7)

    y: Numpy array (labels) (889,1)

    

    Returns:

    train_x: Numpy array containing 80% of random train_data

    test_x: Numpy array containing 20% of random train_data

    train_y: Numpy array containing corresponding labels of train_x

    test_y: Numpy array containing corresponding labels of test_x

    '''

    train_x,test_x,train_y,test_y = train_test_split(X_train,Y_train,test_size=0.2,random_state=42)

    return train_x,test_x,train_y,test_y
from sklearn.base import BaseEstimator,TransformerMixin

from sklearn.preprocessing import MultiLabelBinarizer
class MyLabelBinarizer(TransformerMixin):

    

    def __init__(self,*args,**kwargs):

        self.encoder = MultiLabelBinarizer(*args,**kwargs)

    

    def fit(self,X,y=None):

        return self.encoder.fit(X)

    

    def transform(self,X,y=None):

        return self.encoder.transform(X)
class DataFrameSelector(BaseEstimator,TransformerMixin):

    

    def __init__(self,attribute_names):

        self.attribute_names = attribute_names

        

    def fit(self,X,y=None):

        return self

    

    def transform(self,X,y=None):

        return X[self.attribute_names].values
from sklearn.pipeline import Pipeline,FeatureUnion

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer
def create_pipeline(train_x):

    '''

    Create a single pipeline by cascading 2 pipelines - one for the numerical attributes & one for the categorical text

    attributes.

    

    Argument:

    train_x: Pandas DataFrame object of dimensions (889,7)

    

    Returns:

    full_pipeline: Pipeline object of Scikit learn

    '''

    num_attribs = list(train_x.drop(['Sex','Embarked'],axis=1))

    cat_attribs = ['Sex','Embarked']

    num_pipeline = Pipeline([('selector',DataFrameSelector(num_attribs)),('imputer',SimpleImputer(strategy='median')),

                            ('std_scalar',StandardScaler())])

    cat_pipeline = Pipeline([('selector',DataFrameSelector(cat_attribs)),('my_label_binarizer',MyLabelBinarizer())])

    full_pipeline = FeatureUnion([('num_pipeline',num_pipeline),('cat_pipeline',cat_pipeline)])

    return full_pipeline
def clean_data(train_data,test_data):

    '''

    Remove/Replace NaN entries, remove irrelevant/unusable columns

    

    Arguments:

    train_data: Pandas DataFrame object of dimensions - (891,12)

    test_data: Pandas DataFrame object of dimensions - (418,11)

    

    Returns:

    clean_train_data: Pandas DataFrame object of dimensions - (889,7)

    train_y: Numpy array of dimensions - (889,1)

    clean_test_data: Pandas DataFrame object of dimensions - (418,7)

    '''

    # Clean training data

    clean_train_data = train_data.copy()

    clean_train_data.dropna(subset=['Embarked'],inplace=True)

    train_y = clean_train_data['Survived'].values

    clean_train_data.drop(['PassengerId','Survived','Name','Ticket','Cabin'],axis=1,inplace=True)

    # Clean test data

    clean_test_data = test_data.copy()

    clean_test_data.dropna(subset=['Embarked'],inplace=True)

    clean_test_data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)

    return clean_train_data,train_y,clean_test_data
def preprocess_data(train_data,test_data):

    '''

    Preprocess the data.

    

    Arguments:

    train_data: Pandas DataFrame object, dimension - (891,12)

    test_data: Pandas DataFrame object, dimension - (418,11)

    

    Returns:

    X_train: numpy array - (10,711)

    Y_train: numpy array - (1,711)

    X_cv: numpy array - (10,178)

    Y_cv: numpy array - (1,178)

    X_test: numpy array - (10,418)

    '''

    train_x,train_y,test_x = clean_data(train_data,test_data)

    full_pipeline = create_pipeline(train_x)

    full_pipeline = full_pipeline.fit(train_x)

    X_train = full_pipeline.transform(train_x)

    X_test = full_pipeline.transform(test_x)

    X_train,X_cv,Y_train,Y_cv = split_data(X_train,train_y)

    Y_train,Y_cv = Y_train.reshape(-1,1),Y_cv.reshape(-1,1)

    return X_train.T,Y_train.T,X_cv.T,Y_cv.T,X_test.T
def sigmoid(z):

    '''

    The activation function.

    

    Argument:

    z: Any scalar or numpy array of any size

    

    Returns:

    s: Sigmoid of z

    '''

    s = 1/(1+np.exp(-z))

    return s
def initialize_with_zeros(dim):

    '''

    Creates a vector of zeros of shape (dim,1) for W and initializes b to 0

    

    Argument:

    dim: Number of features in input training sample (nx)

    

    Returns:

    W: zero initialized vector W of shape (dim,1)

    b: zero initialized scalar b (bias)

    '''

    W = np.zeros((dim,1))

    b = 0

    return W,b
def propagate(X,Y,W,b):

    '''

    Computes the cost and gradient in the forward propagation step

    

    Arguments:

    X: Training data of size (nx,m)

    Y: True label vector containing 1 if the passenger drowned and 0 otherwise of size (1,m)

    W: Weights, a numpy array of size (nx,1)

    b: bias, a scalar

    

    Returns:

    cost: negative log-likelihood cost for logistic regression

    dW: gradient of the loss with respect to W, hence same shape as W

    db: gradient of the loss with respect to b. hence same shame as b

    '''

    m = X.shape[1]

    

    Z = np.dot(W.T,X)+b

    A = sigmoid(Z)

    

    cost = -np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))/m

    

    dW = np.dot(X,(A-Y).T)/m

    db = np.sum(A-Y)/m

    

    grads = {'dW':dW,'db':db}

    

    return cost,grads
import matplotlib.pyplot as plt
def optimize(W,b,X,Y,num_iterations,learning_rate,print_cost=False):

    '''

    This function optimizes W and b by running a gradient descent algorithm.

    

    Arguments:

    W: Weights, a numpy array of size (nx,1)

    b: bias, a scalar

    X: Training data of size (nx,m)

    Y: True label vector containing 1 if passenger drowned, 0 otherweise of size (1,m)

    num_iterations: number of iterations of the optimization loop

    learning_rate: learning rate of the gradient descent update rule

    print_cost: True to print the loss every 100 steps

    

    Returns:

    params: dictionary containing weights W and bias b

    grads: dictionary containing the gradients of the loss with respect to weights W and bias b

    cost: list of all the costs computed during the optimization, this will be used to plot the learning curve

    '''

    costs = []

    

    for i in range(num_iterations):

        

        cost,grads = propagate(X,Y,W,b)

        

        dW = grads['dW']

        db = grads['db']

        

        W -= learning_rate*dW

        b -= learning_rate*db

        

        if i%100 == 0:

            costs.append(cost)

            

            if print_cost:

                print('Cost after {}th iteration: {}'.format(i,cost))

    

    costs = np.squeeze(d['costs'])

    plt.plot(costs)

    plt.ylabel('cost',c='w')

    plt.xlabel('iterations (per hundred)',c='w')

    plt.title('Learning rate = {}'.format(learning_rate),c='w')

    plt.show()

        

    params = {'W':W,'b':b}

    

    return params,grads,costs
def predict(W,b,X):

    '''

    Predict whether the label is 0 or 1 using learned logistic regression parameters

    

    Arguments:

    W: Weights, a numpy array of size (nx,1)

    b: bias, a scalar

    X: Training data of size (nx,m)

    

    Returns:

    Y_predictions: a numpy array (vector) containing all predictions (0/1) for the examples in X

    '''

    m = X.shape[1]

    W = W.reshape((X.shape[0],1))

    

    A = sigmoid(np.dot(W.T,X)+b)

    Y_predictions = A>=0.5

    Y_predictions = np.array([[int(val) for val in Y_predictions[0]]])

    

    return Y_predictions
def model(X_train, Y_train, X_cv, Y_cv, X_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):

    """

    Builds the logistic regression model by calling the function implemented previously

    

    Arguments:

    X_train -- training set represented by a numpy array of shape (nx, m_train)

    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)

    X_test -- test set represented by a numpy array of shape (nx, m_test)

    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters

    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize

    print_cost -- Set to true to print the cost every 100 iterations

    

    Returns:

    d -- dictionary containing information about the model.

    """

    W, b = initialize_with_zeros(X_train.shape[0])



    parameters, grads, costs = optimize(W,b,X_train,Y_train,num_iterations,learning_rate,True)



    W = parameters["W"]

    b = parameters["b"]

    

    Y_prediction_test = predict(W, b, X_test)

    Y_prediction_train = predict(W, b, X_train)

    Y_prediction_cv = predict(W, b, X_cv)



    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))

    print('cv accuracy: {}'.format(100 - np.mean(np.abs(Y_prediction_cv-Y_cv))*100))



    d = {"costs": costs,

         "Y_prediction_test": Y_prediction_test, 

         "Y_prediction_train" : Y_prediction_train,

         'Y_prediction_cv' : Y_prediction_cv,

         "W" : W, 

         "b" : b,

         "learning_rate" : learning_rate,

         "num_iterations": num_iterations}

    

    return d
X_train,Y_train,X_cv,Y_cv,X_test = preprocess_data(train_data,test_data)

d = model(X_train, Y_train, X_cv, Y_cv, X_test, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': pd.Series(d['Y_prediction_test'][0])})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")