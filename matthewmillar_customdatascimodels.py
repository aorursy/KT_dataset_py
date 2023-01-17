import numpy as np

import matplotlib.pyplot as plt

from sklearn.base import RegressorMixin

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
# NULL Modle Guses the median for everything so it should be 50/50 

'''

# NOT USEFUL FOR REAL APPLICATIONS EXECPT FOR A BASELINE MODEL

'''

# Custom Sklearn class

class NullRegressor(RegressorMixin):

    def fit(self, X=None, Y=None):

        self.yHat = np.mean(Y) # Makes all predictiions the mean of Y

    

    def predict(self, X=None):

        # Make a array the same number of samples and place the mean of Y n there

        return np.ones(X.shape[0]) * self.yHat  
# Get a regression dataset

from sklearn.datasets import load_boston

X, y = load_boston(return_X_y=True)

print(X.shape)
# Split the data

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.15, random_state = 5)

print(X_train.shape, Y_train.shape)

print(X_test.shape, Y_test.shape)
# Try this ouot

null_model = NullRegressor()

null_model.fit(X_train, Y_train)

y_pred = null_model.predict(X_test)
print(len(Y_test), len(y_pred))
print(null_model.score(X_test, Y_test))
# Cluster modle sing GridSerach and pipline to aid in the traiing

# This will pass the number of cluster from KMeans into a logistic regression model

'''

Pipeline([

    ("sc", StandardScaler()),

    ("km", KMeansSomehow()),

    ("lr", LogisticRegression()

])

'''



from sklearn.base import TransformerMixin

from sklearn.cluster import KMeans

# Used for fitting method

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.datasets import make_blobs

from sklearn.pipeline import Pipeline
class KMeansTransformer(TransformerMixin):

    def __init__(self, *args, **kargs):

        self.model = KMeans(*args, **kargs) # Contrains underlying cluster model



    def fit(self, X):

        self.X = X

        self.model.fit(X)

    

    def fit_transform(self, X, y=None):

        self.fit(X)

        return self.transform(X)

    

    def transform(self, X):

        # To use one hot encoding resahpe to a column vector

        cl = self.model.predict(X).reshape(-1, 1)

        self.oh = OneHotEncoder(categories="auto", sparse=False, drop='first')

        cl_matrix = self.oh.fit_transform(cl)

        return  np.hstack([self.X, cl_matrix])

    
# Cluster data

X, Y = make_blobs(

    n_samples=1000,

    n_features=2,

    centers=3)



# Create pipeline

pipe = Pipeline([("sc", StandardScaler()),

                 ("km", KMeansTransformer()),

                 ("lr", LogisticRegression(penalty="none", solver="lbfgs"))

                ])
# Now fit and score the pipline

pipe.fit(X, Y)

pipe.score(X,Y)