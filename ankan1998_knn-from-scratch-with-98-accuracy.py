import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
class KNN:



    def __init__(self,k=2):

        # Need to specified the number of K default = 2

        self.k=k

    

    def fit(self,X,y):

        # Training on include giving info of training dataset

        self.X_train=X

        self.y_train=y

        

    def predict(self,X):

        y=np.zeros(len(X))

        for idx_of_test, x in enumerate(X): # iterating over test set

            distance_list_of_test_train=[] # empty list to include test_train distance

            for idx_of_train , x_t in enumerate(self.X_train): # iterating over training set data

                difference=x-x_t # next two lines of code calculate distance

                sqrd_dist=difference.dot(difference)

                distance_list_of_test_train.append(sqrd_dist) # appending sqrd distance to distance list

            k_idx=np.argsort(distance_list_of_test_train)[:self.k] #argsort sorts from asc to desc returns index in position

            k_nearest_lbl=[self.y_train[train_y_idx] for train_y_idx in k_idx] #labelling according its index of y_train

            y[idx_of_test]=max(set(k_nearest_lbl), key = k_nearest_lbl.count) # getting max occurences of classes in the list

        return y

    

    def accuracy(self,y_true,y_pred):

        return np.mean(y_true==y_pred)

            

            

        

        
# Classification data

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X=data.data

y=data.target
# Splitting test,train split

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.31, random_state=6)
# Scaling

# It could be done manually, I used sci-kit learn library

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test=sc.fit_transform(X_test)
# As it is either malignant or benign

knnC=KNN(2)
# Fitting the data

knnC.fit(X_train,y_train)

# predicting on data

y=knnC.predict(X_test)

# Calculating Accuracy

knnC.accuracy(y_test,y)