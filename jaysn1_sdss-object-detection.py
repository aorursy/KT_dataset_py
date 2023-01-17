#Importing all the necessary libraries
import pandas as pd
import numpy as np
#Importing downloaded dataset from Kaggle
dataset=pd.read_csv('../input/Skyserver_SQL2_27_2018 6_51_39 PM.csv',skiprows=1)
#Getting the details of the dataset
dataset.info()
#Dropping unimportant fields
dataset=dataset.drop(columns=['objid','specobjid','run','rerun','camcol','field'])
dataset.head()
#For seeing the correlation between the target and the fearures we need to apply label encoder on the class column.
from sklearn.preprocessing import LabelEncoder
dataset=dataset.apply(LabelEncoder().fit_transform)
dataset.corr()
#Getting X and y from the dataset
X=dataset.drop(columns=['class'])
y=dataset.iloc[:,7].values
#Splitting the dataset into train and test data
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8)
#Using KNN algorithm
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)

#Getting the accuracy of the model
accuracy=(y_pred==y_test).sum().astype(float) /len(y_test) *100
accuracy
#Using Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dte=DecisionTreeClassifier()
dte.fit(X_train,y_train)
y_pred=dte.predict(X_test)
#Getting the accuracy of the model
accuracy=(y_pred==y_test).sum().astype(float) /len(y_test) *100
accuracy
#Using Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
y_pred=rfc.predict(X_test)
#Getting the accuracy of the model
accuracy=(y_pred==y_test).sum().astype(float) /len(y_test) *100
accuracy