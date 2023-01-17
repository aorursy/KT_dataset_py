import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from sklearn import tree

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split

print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")
train.shape
train.head()
train["Sex"] = pd.get_dummies(train["Sex"])['female']
train.head()
train["Embarked"] = train["Embarked"].map({'C':0,'Q':1,'S':2})
train["Embarked"] 
train.Embarked.unique()
train.Embarked.fillna(4).unique()
train["Embarked"] = train.Embarked.fillna(4)
train["Embarked"] = train.Embarked.astype('int')
train.head()
train.Age.unique()
train.Age.isnull().sum()
round(train.Age.mean(),3)
train['Age'] = train.Age.fillna(round(train.Age.mean(),2))
train.head()
train.isnull().sum()
train.Cabin.value_counts()
train.dropna().get(['Survived','Cabin'])
train.get(['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
#リスト 6-2-(6)
#  ロジスティック回帰モデル -----------------
def logistic(x, w):
    y = 1 / (1 + np.exp(-((np.hstack((np.ones((x.shape[0],1)),x))).dot(w.T))))
    return y
#リスト 6-2-(9)
# 交差エントロピー誤差 ------------
def cee_logistic(w, x, t):
    X_n = x.shape[0]
    y = logistic(x, w)
    cee = 0
    for n in range(len(y)):
        cee = cee - (t.loc[n] * np.log(y[n]) +
                     (1 - t.loc[n]) * np.log(1 - y[n]))
    cee = cee / X_n
    return cee

# 交差エントロピー誤差の微分 ------------
def dcee_logistic(w, x, t):
    X_n=x.shape[0]
    y = logistic(x, w)
    dcee = np.zeros(8)
    for n in range(len(y)):
        dcee[0] = dcee[0] + (y[n] - t.loc[n])
        for m, key in enumerate(X.keys()):
            dcee[m+1] = dcee[m+1] + (y[n] - t.loc[n]) * x[key][n]
    dcee = dcee / X_n
    return np.array(dcee)
def fit_logistic(w_init, x, t):
    res = minimize(cee_logistic, w_init, args=(x, t),
                   jac=dcee_logistic, method="CG")
    return res.x
W_init = [1, 1, 1, 1, 1, 1, 1, 1]
X = train.get(['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
W = fit_logistic(W_init, X, train['Survived'])
print(W)
print('end')
W
test = pd.read_csv("../input/test.csv")
test.shape
test["Sex"] = pd.get_dummies(test["Sex"])['female']
test["Embarked"] = test["Embarked"].map({'C':0,'Q':1,'S':2})
test["Embarked"] = test.Embarked.fillna(4)
test["Embarked"] = test.Embarked.astype('int')
test['Age'] = test.Age.fillna(round(test.Age.mean(),2))
test_x = test.get(['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
test.head()
test_Y = 1 / (1 + np.exp(-((np.hstack((np.ones((418,1)),test_x))).dot(W.T))))
test_Y
test_Y01 = np.where(test_Y > 0.5, 1, 0)
test_Y01
PassengerId = np.array(test["PassengerId"]).astype(int)
 
my_solution = pd.DataFrame(test_Y01, PassengerId, columns = ["Survived"])
 
my_solution.to_csv("my_test_Y.csv", index_label = ["PassengerId"])
print(os.listdir("."))
X = train.get(['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
y = train['Survived']


clf = LogisticRegression()
skf = StratifiedKFold(shuffle=True)
scoring = {
    'acc': 'accuracy',
    'auc': 'roc_auc',
}
scores = cross_validate(clf, X, y, cv=skf, scoring=scoring)

print('Accuracy (mean):', scores['test_acc'].mean())
print('AUC (mean):', scores['test_auc'].mean())
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

clf = LogisticRegression()
clf.fit(X, y)

print(clf.intercept_)
print(clf.coef_)
t_y = np.dot(clf.coef_,test_x.T) + clf.intercept_
t_y.shape
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
test_Y01 = np.where(sigmoid(t_y) > 0.5, 1, 0)
test_Y01.shape
PassengerId = np.array(test["PassengerId"]).astype(int)
 
my_solution = pd.DataFrame(test_Y01.T, PassengerId, columns = ["Survived"])
 
my_solution.to_csv("my_test_Y.csv", index_label = ["PassengerId"])
