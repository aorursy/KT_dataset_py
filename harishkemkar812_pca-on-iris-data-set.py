from sklearn import datasets

#iris = datasets.load_iris()



#from sklearn 

from sklearn.datasets import load_iris

import numpy as np

import pandas as pd 



from sklearn.model_selection import train_test_split
iris = load_iris()

iris.data.shape

iris.target.shape

iris.feature_names



iris.target_names



iris.target
x = pd.DataFrame(iris.data)

y = iris.target



x.columns = iris.feature_names

x.shape

## 150 rows and 4 columns 

##(150, 4)

y.shape

## 150 rows and 1 column 
x_train,x_test,y_train,y_test =  train_test_split(x,y,test_size = 0.2,random_state = 1)
## Checking tthe shape of train aand test data 

x_train.shape
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()



lr1 = LogisticRegression(penalty='l1')



#A regression model that uses L1 regularization technique is called Lasso Regression and model which uses L2 is called Ridge Regression.

#The key difference between these two is the penalty term.



#The key difference between these techniques is that Lasso shrinks the less important feature’s coefficient to zero thus, removing some feature altogether. So, this works well for feature selection in case we have a huge number of features.
lr.fit(x_train,y_train)
y_predict = lr.predict(x_test)
print(y_predict)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_predict,y_test)
print(accuracy)



## Hence we can see our model created is predicting 83 % correct values 

## Now lets apply PCA and see if accuracy increases 
### Creating PCA such that it will explain 95 % of the variance

from sklearn.decomposition import PCA



sklearn_pca = PCA(n_components = 0.95)



sklearn_pca.fit(x_train)

## Transforming using PCA 



x_train_transformed = sklearn_pca.transform(x_train)

x_test_transformed = sklearn_pca.transform(x_test)
x_train_transformed.shape

x_test_transformed.shape
lr1.fit(x_train_transformed,y_train)
y_predict1 = lr1.predict(x_test_transformed)

accuracy1 = accuracy_score(y_predict1,y_test)
print(accuracy1)



## Thsis is the accuracy of the new model