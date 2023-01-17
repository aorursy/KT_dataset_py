import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')

X = df.iloc[:,:-1].values

y = df.iloc[:,-1].values

X = (X - np.min(X))/(np.max(X)-np.min(X))

X_train, X_test, y_train, y_test = train_test_split(X,y)
def pipeline(X,y,X_test, y_test, alpha, max_iter, bs):

    """

    Sklearn Sanity Check

    """

    print("-"*20,'Sklearn',"-"*20)

    sk = LogisticRegression(penalty='none',max_iter=max_iter)

    sk.fit(X,y)

    sk_y = sk.predict(X_test)

    print('Accuracy ',accuracy_score(y_test, sk_y))

    

    print("-"*20,'Custom',"-"*20)

    me = LogisticReg(learning_rate=alpha,max_iteration=max_iter,batch_size=bs)

    me.fit(X,y)

    yhat = me.predict(X_test)

    me_score = accuracy_score(y_test, yhat)

    print('Accuracy ',me_score)

    me.plot()
class LogisticReg:

    

    def __init__(self, learning_rate=0.01,max_iteration=100, batch_size=10):

        self.learning_rate = learning_rate

        self.max_iter = max_iteration

        self.batch_size = batch_size

        

    def sigmoid(self,x):

        if isinstance(x, np.ndarray):

            result = np.zeros((x.shape[0]))

            for i in range(x.shape[0]):

                result[i] = np.exp(x[i]) / (1+ np.exp(x[i]))

            return result

        else:

            return np.exp(x) / (1+ np.exp(x))

    

    def sgd(self, X, y):

        n_samples, n_features = X.shape

        self.betas = np.zeros(n_features) # column vector, parameters

        costs = []

        for it in range(self.max_iter):

            indices = np.random.randint(0,X.shape[0],size=self.batch_size)



#             ----------------------

#                 Forward Pass

#             ----------------------

            prediction = self.sigmoid(np.dot(X[indices, :],self.betas))

            

            error = y[indices] - prediction

#             ----------------------

#                 Backward Pass

#             ----------------------

            cost = (-1 / indices.shape[0]) * (y[indices] @ np.log(prediction) + (1 - y[indices]) @ np.log(1-prediction) )

            gradient = (1 / indices.shape[0]) * (X[indices, :].T @ error)

        

            self.betas = self.betas - (self.learning_rate * -gradient)

            costs.append(cost)

            

            if it % (self.max_iter / 10)==0:

                accuracy = accuracy_score(y[indices],np.round(prediction))

                print(f"iteration: {it}, Cost: {cost}, Accuracy: {accuracy}")

            

        self.history = costs

            

        

    def plot(self):

        fig, ax = plt.subplots(1,1,figsize=(20,10),facecolor='white')

        ax.plot(range(self.max_iter),self.history)

        plt.show()

        

    def fit(self, X, y):

        """

        Fit logistic model using Stochastic Gradient Descent

        """

        print(X.shape)

        X = np.insert(X,0,1,axis=1) # add 1s for matrix multiplication

        

        self.sgd(X,y)

    

    def predict(self, X):

        X = np.insert(X,0,1,axis=1)

        yhat = np.dot(X,self.betas)

        yhat = self.sigmoid(yhat)

        return np.round(yhat)

    

    def score(self, X,y):

        yhat = self.predict(X)

        return accuracy_score(y,yhat)
pipeline(X_train, y_train, X_test, y_test, alpha=2,max_iter=1000, bs=250)
class LRGD:

    

    def __init__(self, learning_rate=0.01,max_iteration=100, batch_size=10):

        self.learning_rate = learning_rate

        self.max_iter = max_iteration

        self.batch_size = batch_size

        

    def sigmoid(self,x):

        if isinstance(x, np.ndarray):

            result = np.zeros((x.shape[0]))

            for i in range(x.shape[0]):

                result[i] = np.exp(x[i]) / (1+ np.exp(x[i]))

            return result

        else:

            return np.exp(x) / (1+ np.exp(x))

    

    def gd(self, X, y):

        n_samples, n_features = X.shape

        self.betas = np.zeros(n_features) # column vector, parameters

        costs = []

        for it in range(self.max_iter):

            

            indices = np.arange(0,n_samples,1)

#             indices = np.random.randint(0,X.shape[0],size=self.batch_size)



#             ----------------------

#                 Forward Pass

#             ----------------------

            prediction = self.sigmoid(np.dot(X[indices, :],self.betas))

            

            error = y[indices] - prediction

#             ----------------------

#                 Backward Pass

#             ----------------------

            cost = (-1 / indices.shape[0]) * (y[indices] @ np.log(prediction) + (1 - y[indices]) @ np.log(1-prediction) )

            gradient = (1 / indices.shape[0]) * (X[indices, :].T @ error)

        

            self.betas = self.betas - (self.learning_rate * -gradient)

            costs.append(cost)

            

            if it % (self.max_iter / 10)==0:

                accuracy = accuracy_score(y[indices],np.round(prediction))

                print(f"iteration: {it}, Cost: {cost}, Accuracy: {accuracy}")

            

        self.history = costs

            

        

    def plot(self):

        fig, ax = plt.subplots(1,1,figsize=(20,10),facecolor='white')

        ax.plot(range(self.max_iter),self.history)

        plt.show()

        

    def fit(self, X, y):

        """

        Fit logistic model using Stochastic Gradient Descent

        """

        X = np.insert(X,0,1,axis=1) # add 1s for matrix multiplication

        self.gd(X,y)

    

    def predict(self, X):

        X = np.insert(X,0,1,axis=1)

        yhat = np.dot(X,self.betas)

        yhat = self.sigmoid(yhat)

        return np.round(yhat)

    

    def score(self, X,y):

        yhat = self.predict(X)

        return accuracy_score(y,yhat)
wgd = LRGD(learning_rate=2, max_iteration=5000, batch_size=50)

wgd.fit(X_train,y_train)

print(wgd.score(X_test, y_test))

wgd.plot()
sk = LogisticRegression(max_iter=1000)

sk.fit(X_train, y_train)

sk.score(X_test,y_test)