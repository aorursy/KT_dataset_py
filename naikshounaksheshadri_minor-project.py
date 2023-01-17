# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

%matplotlib inline



import sklearn 



import seaborn as sns



from scipy import optimize

import sklearn



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/minor-project-2020/train.csv")

df=df.drop(['id'],axis=1)

df







from sklearn.model_selection import train_test_split



X=df.iloc[:,0:88]

Y=df.iloc[:,88]



X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=121)



#corr=X.corr()

#mask = np.zeros_like(corr, dtype=np.bool)

#mask[np.triu_indices_from(mask)] = True



#f, ax = plt.subplots(figsize=(11, 9))

#cmap = sns.diverging_palette(220, 10, as_cmap=True)





#sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,

            #square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)



#plt.show()

m_train, n_train = X_train.shape

m_val, n_val = X_val.shape



initial_theta = np.zeros(n_train+1)



# Add intercept term to X

X_train = np.concatenate([np.ones((m_train, 1)), X_train], axis=1)

X_val = np.concatenate([np.ones((m_val, 1)), X_val], axis=1)



def sigmoid(z):

    """

    Compute sigmoid function given the input z.

    

    Parameters

    ----------

    z : array_like

        The input to the sigmoid function. This can be a 1-D vector 

        or a 2-D matrix. 

    

    Returns

    -------

    g : array_like

        The computed sigmoid function. g has the same shape as z, since

        the sigmoid is computed element-wise on z.

        

    Instructions

    ------------

    Compute the sigmoid of each value of z (z can be a matrix, vector or scalar).

    """

   

    z = np.array(z)

    

    

    g = np.zeros(z.shape)



  



    g = 1 / (1 + np.exp(-z))



    return g
def costFunction(theta, X, y):

    """

    Compute cost and gradient for logistic regression. 

    

    Parameters

    ----------

    theta : array_like

        The parameters for logistic regression. This a vector

        of shape (n+1, ).

    

    X : array_like

        The input dataset of shape (m x n+1) where m is the total number

        of data points and n is the number of features. We assume the 

        intercept has already been added to the input.

    

    y : array_like

        Labels for the input. This is a vector of shape (m, ).

    

    Returns

    -------

    J : float

        The computed value for the cost function. 

    

    grad : array_like

        A vector of shape (n+1, ) which is the gradient of the cost

        function with respect to theta, at the current values of theta.

        

    Instructions

    ------------

    Compute the cost of a particular choice of theta. You should set J to 

    the cost. Compute the partial derivatives and set grad to the partial

    derivatives of the cost w.r.t. each parameter in theta.

    """

    

    m = y.size  # number of training examples



   

    J = 0

    grad = np.zeros(theta.shape)



  

    h = sigmoid(X.dot(theta.T))

    

    J = (1 / m) * np.sum(-y.dot(np.log(h)) - (1 - y).dot(np.log(1 - h)))

    grad = (1 / m) * (h - y).dot(X)

    

    

    return J, grad
# set options for optimize.minimize

options= {'maxiter': 400}

m,n=X_train.shape





res = optimize.minimize(costFunction,

                        initial_theta,

                        (X_train, Y_train),

                        jac=True,

                        method='TNC',

                        options=options)





cost = res.fun





theta = res.x





print('Cost at theta found by optimize.minimize: {:.3f}'.format(cost))
def predict(theta, X):

    """

    Predict whether the label is 0 or 1 using learned logistic regression.

    Computes the predictions for X using a threshold at 0.5 

    (i.e., if sigmoid(theta.T*x) >= 0.5, predict 1)

    

    Parameters

    ----------

    theta : array_like

        Parameters for logistic regression. A vecotor of shape (n+1, ).

    

    X : array_like

        The data to use for computing predictions. The rows is the number 

        of points to compute predictions, and columns is the number of

        features.



    Returns

    -------

    p : array_like

        Predictions and 0 or 1 for each row in X. 

    

    Instructions

    ------------

    Complete the following code to make predictions using your learned 

    logistic regression parameters.You should set p to a vector of 0's and 1's    

    """

    

    m = X.shape[0] # Number of training examples



    # You need to return the following variables correctly

    p = np.zeros(m)



   



    p = np.round(sigmoid(X.dot(theta.T)))

  

    return p
p = predict(theta, X_train)

print('Train Accuracy: {:.2f} %'.format(np.mean(p == Y_train) * 100))

p = predict(theta, X_val)

print('Val Accuracy: {:.2f} %'.format(np.mean(p == Y_val) * 100))
sklearn.metrics.roc_auc_score(Y_val, p)
df_test=pd.read_csv("/kaggle/input/minor-project-2020/test.csv")
id=df_test.iloc[:,0]

X_test=df_test.iloc[:,1:89]

m_test,n_test=X_test.shape

print(X_test.shape)

X_test = np.concatenate([np.ones((m_test, 1)), X_test], axis=1)

print(X_test.shape)

p_test=predict(theta,X_test)
my_submission = pd.DataFrame({'Id': id, 'target': p_test})

my_submission.to_csv('submission.csv', index=False)