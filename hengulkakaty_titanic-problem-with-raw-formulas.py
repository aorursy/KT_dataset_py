# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd
df1 = pd.read_csv('../input/train.csv')

df2 = pd.read_csv('../input/test.csv')
df1["Sex"] = df1["Sex"].replace('male', 0)

df1["Sex"] = df1["Sex"].replace('female', 1)



df2["Sex"] = df2["Sex"].replace('male', 0)

df2["Sex"] = df2["Sex"].replace('female', 1)



df1["Age"] = df1["Age"].fillna(df1["Age"].dropna().median())

df2["Age"] = df2["Age"].fillna(df1["Age"].dropna().median())
df1.loc[ df1['Age'] <= 16, 'Age'] = 0,

df1.loc[(df1['Age'] > 16) & (df1['Age'] <= 26), 'Age'] = 1,

df1.loc[(df1['Age'] > 26) & (df1['Age'] <= 36), 'Age'] = 2,

df1.loc[(df1['Age'] > 36) & (df1['Age'] <= 62), 'Age'] = 3,

df1.loc[ df1['Age'] > 62, 'Age'] = 4



df2.loc[ df2['Age'] <= 16, 'Age'] = 0,

df2.loc[(df2['Age'] > 16) & (df2['Age'] <= 26), 'Age'] = 1,

df2.loc[(df2['Age'] > 26) & (df2['Age'] <= 36), 'Age'] = 2,

df2.loc[(df2['Age'] > 36) & (df2['Age'] <= 62), 'Age'] = 3,

df2.loc[ df2['Age'] > 62, 'Age'] = 4
df1["Fare"] = df1["Fare"].fillna( df1["Fare"].dropna().median() )

df2["Fare"] = df2["Fare"].fillna( df2["Fare"].dropna().median() )



df1.loc[ df1['Fare'] <= 17, 'Fare'] = 0,

df1.loc[(df1['Fare'] > 17) & (df1['Fare'] <= 30), 'Fare'] = 1,

df1.loc[(df1['Fare'] > 30) & (df1['Fare'] <= 100), 'Fare'] = 2,

df1.loc[ df1['Fare'] > 100, 'Fare'] = 3



df2.loc[ df2['Fare'] <= 17, 'Fare'] = 0,

df2.loc[(df2['Fare'] > 17) & (df2['Fare'] <= 30), 'Fare'] = 1,

df2.loc[(df2['Fare'] > 30) & (df2['Fare'] <= 100), 'Fare'] = 2,

df2.loc[ df2['Fare'] > 100, 'Fare'] = 3


df1["Embarked"] = df1["Embarked"].fillna("C") 

df1["Embarked"] = df1["Embarked"].replace("S", 0)

df1["Embarked"] = df1["Embarked"].replace("C", 1)

df1["Embarked"] = df1["Embarked"].replace("Q", 2)



df2["Embarked"] = df2["Embarked"].fillna("C") 

df2["Embarked"] = df2["Embarked"].replace("S", 0)

df2["Embarked"] = df2["Embarked"].replace("C", 1)

df2["Embarked"] = df2["Embarked"].replace("Q", 2)
df1["Family Size"] = df1["SibSp"]  + df1["Parch"] + 1

df2["Family Size"] = df2["SibSp"]  + df2["Parch"] + 1
df1['Title'] = df1['Name'].str.extract('([A-Za-z]+)\.', expand = False)

df2['Title'] = df2['Name'].str.extract('([A-Za-z]+)\.', expand = False)



df1['Title'].value_counts()

df2['Title'].value_counts()



title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 

                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,

                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }



df1['Title'] = df1['Title'].map(title_mapping)

df2['Title'] = df2['Title'].map(title_mapping)
split = 600



data = np.array(df1)  

data_2 = np.array(df2)

np.set_printoptions(suppress=True)
#   0           1             2      3       4     5     6        7      8       9       10       11

#PassengerId	Survived	Pclass	Name	Sex	  Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked



X_train = np.asarray( data[0:split, [2, 4, 5, 9, 11, 12, 13] ], dtype=np.float32 )

y_train = np.asarray( data[0:split, 1:2], dtype=np.float32)



X_val = np.asarray( data[split:891, [2, 4, 5, 9, 11, 12, 13] ], dtype=np.float32)

y_val = np.asarray( data[split:891, 1:2], dtype=np.float32)
#   0           1         2      3     4      5       6       7        8      9       10

#PassengerId	Pclass	Name	Sex	  Age	SibSp	Parch	Ticket	 Fare	Cabin	Embarked



X_test = np.asarray( data_2[:, [1, 3, 4, 8, 10, 11, 12] ], dtype=np.float32)
global m

global n    

m,n = X_train.shape 
def featurenormal(X):

    mu = np.mean(X, axis = 0)

    sigma = np.std(X, axis = 0)

    X = (X - mu) / sigma

    return X
X_train = featurenormal(X_train)

X_val = featurenormal(X_val)

X_test = featurenormal(X_test)
ones = np.ones((X_train[:,0:1].shape))

X_train = np.hstack([ones, X_train])



ones_1 = np.ones((X_val[:,0:1].shape))

X_val = np.hstack([ones_1,X_val])



ones_2 = np.ones((X_test[:,0:1].shape))

X_test = np.hstack([ones_2, X_test])

initial_theta = np.zeros((n+1,1)) 
#z = 0

def sigmoid(z):

    z = 1 / (1 + np.exp(-z))

    return z

global lam

lam = .06

def costfunction(initial_theta, X_train, y_train, lam):

    

    #lam = 0

    h = sigmoid(X_train.dot(initial_theta))

    l = np.log(h)

    l1 = np.log(1-h)

    J = -(1/m) * ((y_train.T.dot(l)) + ((1-y_train).T.dot(l1))) + (lam/(2*m) * (initial_theta[1:].T.dot(initial_theta[1:])) )

    

    #grad = ((1/m) * (X_train.T.dot(h-y_train)))  +  ((lam/m)*initial_theta)

    J= J.flatten()

    #grad=grad.flatten()

    return J
J = costfunction(initial_theta, X_train, y_train, lam)
import scipy.optimize as opt

final_theta  = opt.fmin( costfunction, x0=initial_theta, args=(X_train, y_train, lam), maxiter=10000, full_output=True)
print('Cost with Theta [0,0,0] is :', J)

#print('Gradient with Theta [0,0,0] is :', grad)
print("Final theta found by minimization function is :", final_theta[0])

y_pred_test = sigmoid(X_test.dot(final_theta[0].T))

y_pred_test = y_pred_test.reshape(418,1)

y_pred_test = np.round(y_pred_test, decimals=0)



y_trainpred = sigmoid(X_train.dot(final_theta[0].T))

y_trainpred = y_trainpred.reshape(split,1)

y_trainpred = np.round(y_trainpred)



Accuracy = (np.sum(y_trainpred == y_train) / y_trainpred.size) * 100

print('Training set accuracy is ', Accuracy)



t = 891 - split

y_valpred = sigmoid(X_val.dot(final_theta[0].T))

y_valpred = y_valpred.reshape(t,1)

y_valpred = np.round(y_valpred)



Accuracy_val = (np.sum(y_valpred == y_val) / y_valpred.size) * 100

print('Validation set accuracy is ', Accuracy_val)
prediction = pd.read_csv("../input/gender_submission.csv")
#np.argmax(y_pred_test, axis = 1)

prediction['Survived'] = np.int64(y_pred_test)

prediction.to_csv('submission.csv', index = False)