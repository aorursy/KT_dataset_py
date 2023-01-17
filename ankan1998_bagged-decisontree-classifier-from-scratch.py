import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.tree import DecisionTreeClassifier
class BaggedDecisonTreeClassifier:

    

    def __init__(self,num_of_bagged=5):

        # Initialised with number of bagged models

        self.num_of_bagged=num_of_bagged

        

    def fit(self,X,y):

        # to store the models

        self.models=[]

        for i in range(self.num_of_bagged):

            indexs=np.random.choice(len(X),size=len(X))# sample with replacement

            Xi=X[indexs]# Chossing random samples

            Yi=y[indexs]

            # Training for each sample bunch by Decision Tree Classifier

            model=DecisionTreeClassifier()

            model.fit(Xi,Yi)

            # Storing the models

            self.models.append(model)

            

    def predict(self,X):

        pred=np.zeros(len(X))

        # predicting with each stored models

        for model in self.models:

            pred=pred+model.predict(X)

        return np.round(pred/self.num_of_bagged) # Model averaging

    

    def acc(self,y_true,y_pred):

        return np.mean(y_true==y_pred)

        
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X=data.data

y=data.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)
# Scaling

# It could be done manually, I used sci-kit learn library

#from sklearn.preprocessing import StandardScaler

#sc = StandardScaler()

#X_train = sc.fit_transform(X_train)

#X_test=sc.fit_transform(X_test)
# Calling with 10 Decision Trees

bdtc=BaggedDecisonTreeClassifier(10)
# Fitting the model

bdtc.fit(X_train,y_train)
# Predicting with model

y_pred=bdtc.predict(X_test)
# Calculating Accuracy

bdtc.acc(y_test,y_pred)