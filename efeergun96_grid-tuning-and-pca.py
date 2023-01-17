#importing the required libraries

import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore")
# exploring the data

data = pd.read_csv("../input/data.csv")

data.head()
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)   # dropping some un important fields 

data.info()
data.describe()
labeller = LabelEncoder()
labeller.fit(data.diagnosis)
data.diagnosis = labeller.transform(data.diagnosis)
y = data.diagnosis.values
data.drop(["diagnosis"],axis=1,inplace=True)
x = data.values
pca = PCA(n_components=2,whiten=True)
pca.fit(x)
x_new = pca.transform(x)

print("variance ratio: ", pca.explained_variance_ratio_)
print("sum of variance percentage: ", sum(pca.explained_variance_ratio_))
train_x, test_x, train_y, test_y = train_test_split(x_new,y,test_size=0.2,random_state=99)
# default Logistic Regression
logreg_default = LogisticRegression()
logreg_default.fit(train_x,train_y)
print("Default Real Test Score: ",logreg_default.score(test_x,test_y))
# GridSearch implemented on some fiels
parameters = {"C":np.arange(0.1,2.0,0.1),"penalty":("l2","l1")
             ,"max_iter":np.arange(100,2000,100)}
logreg_grid = LogisticRegression()

logreg_cv = GridSearchCV(logreg_grid,parameters)
logreg_cv.fit(train_x,train_y)
print("tuned hyperparameters: ",logreg_cv.best_params_)
print("tuned highest score: ",logreg_cv.best_score_)

logreg_grid = LogisticRegression(C=0.2,penalty="l1",max_iter=100)
logreg_grid.fit(train_x,train_y)
print("Tuned Real Test Result: ",logreg_grid.score(test_x,test_y))
# default KNN
knn_default = KNeighborsClassifier()
knn_default.fit(train_x,train_y)
print("Default Real Test Score: ",knn_default.score(test_x,test_y))
# GridSearch implemented KNN
parameters = {"algorithm":("auto", "ball_tree", "kd_tree", "brute"), "n_neighbors":np.arange(1,20),"p":(2,1)}
knn_grid = KNeighborsClassifier()

knn_cv = GridSearchCV(estimator=knn_grid,param_grid=parameters)
knn_cv.fit(train_x,train_y)
print("tuned hyperparameters: ",knn_cv.best_params_)
print("tuned highest score: ",knn_cv.best_score_)

knn_grid = KNeighborsClassifier(algorithm="auto",n_neighbors=7,p=1)
knn_grid.fit(train_x,train_y)
print("Tuned Real Test Result: ",knn_grid.score(test_x,test_y))
# default SVC
svc_default = SVC()
svc_default.fit(train_x,train_y)
print("Default Real Test Score: ",svc_default.score(test_x,test_y))
# GridSearch implemented Support Vector
parameters = {"C":np.arange(0.1,2.0,0.1),"kernel":("linear", "poly", "rbf", "sigmoid"),"probability":(False,True)}
svc_grid = SVC()

svc_cv = GridSearchCV(estimator=svc_grid,param_grid=parameters)
svc_cv.fit(train_x,train_y)
print("tuned hyperparameters: ",svc_cv.best_params_)
print("tuned highest score: ",svc_cv.best_score_)
svc_grid = SVC(C=0.9,kernel="rbf",probability=False)
svc_grid.fit(train_x,train_y)
print("Tuned Real Test Result: ",svc_grid.score(test_x,test_y))
