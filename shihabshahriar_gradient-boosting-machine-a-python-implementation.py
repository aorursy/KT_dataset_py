import numpy as np 

import pandas as pd

np.random.seed(1539)

from sklearn.datasets import load_boston,load_diabetes

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error,mean_absolute_error
"""

data = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")

features,target = data.values[:,:-1],data.values[:,-1] 

"""

#features,target = load_boston(return_X_y=True)



features,target = load_diabetes(return_X_y=True)

features.shape,target.shape
%%time

K = 100

avg_err = []

for i in range(K):

    X,Xtest,Y,Ytest = train_test_split(features,target,test_size=.20)

    

    lr = .1  # learning rate

    T = 100  # Number of rounds

    models = []

    #-------------TRAINING----------------

    Ygrad = Y

    for t in range(T):

        reg = DecisionTreeRegressor(max_depth=2).fit(X,Ygrad)

        yp = reg.predict(X)

        Ygrad = Ygrad - lr*yp

        models.append(reg)



    #-------------PREDICTING-------------

    N = len(Xtest)

    Y_pred = np.full((N,),0,dtype='float')

    for reg in models:

        Y_pred += lr*reg.predict(Xtest)

    avg_err.append(mean_absolute_error(Ytest,Y_pred))

    

print("Mean Abs Error:",sum(avg_err)/len(avg_err))
%%time

K = 100

avg_err = []

for i in range(K):

    X,Xtest,Y,Ytest = train_test_split(features,target,test_size=.20)

    reg = GradientBoostingRegressor(n_estimators=100,learning_rate=0.1).fit(X,Y) #DecisionTreeRegressor().fit(X,Y)

    Y_pred = reg.predict(Xtest)

    avg_err.append(mean_absolute_error(Ytest,Y_pred))

    

print("Mean Abs Error:",sum(avg_err)/len(avg_err))























