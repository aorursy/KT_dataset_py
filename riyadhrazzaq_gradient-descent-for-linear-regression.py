import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



from sklearn.linear_model import SGDRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score
class LinearRegress:

    

    """

    Linear Regression with Gradient Descent.

    """

    def __init__(self,bias=True, learning_rate = 0.01, max_iteration=1000, keep_hist = True):

        """

        Linear Regression with Gradient Descent.

        

        params

        ------

        bias: bool. Default=True. Whether to add bias variable or not.

        

        learning_rate: float. learning rate for gradient descent. 

        

        max_iteration: scalar. Maximum iteration for Gradient Descent.

        

        keep_hist: bool. If true, stores J and W for each iteration.

        

        """

        self.bias = bias

        self.w = None

        self.error = None

        self.alpha = learning_rate

        self.max_iter = max_iteration

        self.n_samples = None

        self.n_features = None

        self.keep_hist = keep_hist

        self.history = []

        

    def gradient_descent(self,X,y):

        """

        Simple SGD. 



        params

        ------

        X: Predictors. Shape(n_samples, n_features)



        y: Target. Shape(n_samples)



        """

        n_samples = len(y)

        

        # inserts column of 1s for bias.

        X = np.insert(X,0,1,axis=1) 

        

        # Weight Matrix. Shape(n_features,1)

        w = np.random.rand(X.shape[1])

        

        for itr in range(self.max_iter):

            # forward pass

            yhat = np.dot(X,w) # (n_samples, 1)

            # objective

            error = yhat - y

            J = (1/(2 * n_samples)) * np.dot(error,error)

            

            # backward pass

            w = w - (self.alpha / n_samples) * np.dot(X.T,error)

            

            if self.keep_hist is True:

                self.history.append((J,w))

            if self.verbose and (itr % 100 == 0 or itr == self.max_iter - 1):

                print("Iteration: %d J: %.2f" % (itr,J))

        self.w = w



    def fit(self,X,y,verbose=True):

        """

        Train linear regression model.

        """

        self.verbose = verbose

        self.gradient_descent(X,y)



    def score(self,X,y):

        """

        Return the coefficient of determination R^2 of the prediction.

        

        params

        ------

        X: array-like of shape (n_samples, n_features)

            Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic objects instead, shape = (n_samples, n_samples_fitted), where n_samples_fitted is the number of samples used in the fitting for the estimator.

            

        y: array-like of shape (n_samples,)

            True values for X.

        

        returns

        -------

        score: float.

            R^2 score.

        """

        yhat = self.predict(X)

        return r2_score(y,yhat)

    

    def predict(self,X):

        """

        Predicts using linear model.

        

        params

        ------

        X: array-like of shape (n_samples, n_features)

            Test samples.

        returns

        -------

        y: array-like of shape(n_samples,)

            Output of the model.

        """

        X = np.insert(X,0,1,axis=1)

        yhat = np.dot(X,self.w)

        return yhat
# heart data

df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

X = df.iloc[:,:-1].copy()

y = df.iloc[:,-1].copy()

X_train,X_test,y_train,y_test = train_test_split(X.values,y.values)
model = LinearRegress(learning_rate=0.000001,max_iteration=1000)

model.fit(X_train,y_train,verbose=True)

model.score(X_test,y_test)
fig, ax = plt.subplots(facecolor='white')

costs = [tpl[0] for tpl in model.history]

coefs = [tpl[1] for tpl in model.history]

ax.plot(range(len(costs)),costs,color='green')

ax.set_xlabel('Iteration')

ax.set_ylabel('Objective')

plt.show()
sk = SGDRegressor()

sk.fit(X_train,y_train)

sk.score(X_test,y_test) > model.score(X_test,y_test)