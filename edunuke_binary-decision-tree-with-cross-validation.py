import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.utils import shuffle

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_auc_score
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

df = pd.read_csv("../input/creditcard.csv")

df =  shuffle(df).reset_index(drop=True)

df.head()
frac =0.80 

X_train = df.sample(frac=frac)

count_FRAUDE = len(X_train)



#X_test contains all the transaction not in X_train.

X_test = df.loc[~df.index.isin(X_train.index)].reset_index(drop=True).astype(float)



#Add our target features to y_train and y_test.

y_train = pd.DataFrame({'Class':X_train.Class.reset_index(drop=True)})







y_test = pd.DataFrame({'Class':X_test.Class.reset_index(drop=True)})



#Drop target features from predictors X_train and X_test.

X_train = X_train.drop(['Class'], axis = 1).reset_index(drop=True)

X_test = X_test.drop(['Class'], axis = 1).reset_index(drop=True)

features = X_train.columns.values

for feature in features:

    mean, std = df[feature].mean(), df[feature].std()

    X_train.loc[:, feature] = (X_train[feature] - mean) / std

    X_test.loc[:, feature] = (X_test[feature] - mean) / std

nrow = len(X_train)

depthList = np.array(range(1,20)) #search depth list

xvalMSE =[]

epoch = 1

score_history =[]

for iDepth in depthList:



    for ixval in range(epoch):



        xTrain = X_train.as_matrix()

        yTrain = y_train.as_matrix().ravel()

        xTest = X_test.as_matrix()

        yTest = y_test.as_matrix().ravel()



    

        treeModel = DecisionTreeClassifier(max_depth = iDepth)

        treeModel.fit(xTrain, yTrain)

        

        treePrediction = treeModel.predict(xTest)

        treePrediction=np.array(treePrediction,dtype=np.float32)

        yTest = np.array(yTest,dtype=np.float32)

        

        error = [yTest[r] - treePrediction[r] for r in range(len(yTest))]



        

        if ixval == 0:

            oosErrors = sum([e*e for e in error])

        else:

            oosErrors += sum([e*e for e in error])

    score = roc_auc_score(yTest, treePrediction)

    score_history.append(score)    

    mse = oosErrors/nrow

    xvalMSE.append(mse)
plt.figure()

plt.plot(depthList,xvalMSE)

plt.axis("tight")

plt.xlabel("Tree Depth")

plt.ylabel("MSE")

opt_depth = depthList[np.argmin(xvalMSE)]

plt.plot(opt_depth,np.amin(xvalMSE),"or")
plt.figure()

plt.plot(depthList,score_history)

plt.axis("tight")

plt.xlabel("Tree Depth")

plt.ylabel("ROC AUC Scores")

plt.show()
print("Minimum MSE:",np.amin(xvalMSE))

print("Maximum ROC AUC Score:", np.amax(score_history))