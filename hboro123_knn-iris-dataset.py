## Importing libraries and iris datset



from sklearn.datasets import load_iris

from sklearn.neighbors import KNeighborsClassifier  ## class of sklearn which gives classify the dataset

from sklearn.linear_model import LogisticRegression ## Importing logistic regression model from sklearn

import numpy as np

##Evaluation metrics for KNN



from sklearn import metrics

## save iris datset into an object

iris= load_iris()

type(iris)
## print out the dataset



print(iris)
## print out the feature names or columns names or independent variables



print(iris.feature_names)
## print out the response name or dependent variable or target columns. It will print out encoded names.



print(iris.target_names)
## print out the integer of dependent varaibles



print(iris.target)
## print out the dimension of data(1st dimension= no. of obs, 2nd dimension= no. of features)



print(iris.data.shape)
## print out the dimension of target columns/dependent variable



print(iris.target.shape)
## storing features in x object and target in y object



x=iris.data

y=iris.target
## 1. instantiate the estimator



knn= KNeighborsClassifier(n_neighbors=1)  ### n_neighbours=1 will look for 1 similar group
## 2. To look inside the model and its parameters



print(knn)
## 3. Fit the data into model



knn.fit(x,y)
## 4. Predict the unknown data for single obs



knn.predict([[3,5,3,2]])  ## keep in mind in latest scikit learn to give an unknown list use double brackets



##For this it has predicted sertosa

## 5. Predict the unknown dat of multiple obs



knn.predict([[3,5,6,7],[7,6,9,3]])



## for this it has predicted virginica
## instansiate the model

knn= KNeighborsClassifier(n_neighbors=5)
## fit the data into model



knn.fit(x,y)
##predict the depenent values for KNN=5



knn_predict=knn.predict(x)

## predict the unknown values



y_new=[[3, 5, 4, 2], [5, 4, 3, 2]]



knn_predict=knn.predict(y_new)

print(knn_predict)



## for this it has predicted versicolor
## using logistic regression for same problem and identify the accuracy of its model
## instansiate the model



logreg= LogisticRegression(max_iter=1000)

print(logreg)
##fit the model into dataset



logreg.fit(x,y)
## predict the dependent values



logreg_predict= logreg.predict(x)
## Predict the unknown values



log_predict=logreg.predict([[3,5,3,1]])

print(log_predict)



## logistic regression predcit the specis for given unknown value
##checking accuracy of logistic regression model



print(metrics.accuracy_score(y,logreg_predict))



from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4,random_state=4)



## fitting test/train dataset into knn  , k=5



knn.fit(x_train,y_train)
## predicting the dataset

y_pred_tt= knn.predict(x_test)
## checking the accuracy of dataset

print(metrics.accuracy_score(y_test,y_pred_tt))
log_tt= LogisticRegression()

log_tt.fit(x_train,y_train)
y_log_tt=log_tt.predict(x_test)
print(metrics.accuracy_score(y_test,y_log_tt))