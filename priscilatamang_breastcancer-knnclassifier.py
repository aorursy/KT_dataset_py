# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")
data.head()
data.shape
data.info()
data.isnull().sum() #To check nno of null values present in each column
data.drop(columns=['Unnamed: 32'], inplace=True) #Removing "Unnamed: 32" becasue of too many NULL VALUES.
data.isnull().sum()
data
data['diagnosis'].value_counts() #Checking the no of M and B of diagnosis column
#Replacing M and B with 1 and 0 respectively
data['diagnosis'].replace('M','1',inplace=True)
data['diagnosis'].replace('B','0',inplace=True)
data
#Extracting X
X=data.iloc[:,2:].values
print("The shape of X:",X.shape)

#Extracting Y
Y=data.iloc[:,1].values
print("The shape of Y:",Y.shape)
#Scaling
scaler=StandardScaler()
X=scaler.fit_transform(X)

#Splitting
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)

print("The shape of X_train:",X_train.shape)
print("The shape of Y_train:",Y_train.shape)
print(Y_test.shape)
np.sqrt(X_train.shape[0])
classifier=DecisionTreeClassifier(criterion='entropy', max_depth=8, max_features=9, max_leaf_nodes=5, random_state=6)
classifier.fit(X_train, Y_train)
#Predicting
Y_predict=classifier.predict(X_test)
print(Y_predict.shape)

#Accuracy
AS1=accuracy_score(Y_test,Y_predict) 
print("The accuracy score using decision tree classifier:", AS1)
#Creating Variable
param_dist={"criterion":["gini","entropy"],"max_depth":[1,2,3,4,5,6,7,None],"max_features":[1,2,3,4,5,6,7,None],"random_state":[0,1,2,3,4,5,6,7,8,9,None],"max_leaf_nodes":[0,1,2,3,4,5,6,7,8,9,None]}

#Applying Grid-Search-CV
grid=GridSearchCV(classifier, param_grid=param_dist, cv=10, n_jobs=-1)

#Training the model after applying Grid-Search-CV
grid.fit(X_train,Y_train)
OHV=grid.best_params_ 
print("The values of Optimal Hyperparameters are",OHV)
Acc1=grid.best_score_
print("The Accuracy Score is",Acc1)
print("Accuracy using DecisionTreeClassifier:", Acc1*100,"%")
knn = KNeighborsClassifier(n_neighbors =13)
knn.fit(X_train,Y_train)

#Predicting
Y_predict=knn.predict(X_test) 
print(Y_predict) 

#Finding Accuracy
AS2=accuracy_score(Y_test,Y_predict) 
print("The accuracy score using knn:", AS2)
accuracy=[]
for i in range (1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,Y_train)
    Y_predict = knn.predict(X_test)
    accuracy.append(metrics.accuracy_score(Y_test, Y_predict))

#Plotting graph
plt.plot(range(1,40), accuracy)
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(X_train, Y_train)
Y_predict = knn.predict(X_test)
Acc2=accuracy_score(Y_test,Y_predict) 
print("The accuracy score :", Acc2)
print("Accuracy using knn Classifier:", Acc2*100,"%")