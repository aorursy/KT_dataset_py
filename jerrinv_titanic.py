# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib 
import matplotlib.pyplot as plt
from IPython.display import HTML,SVG
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
# used to ignore warnings
import warnings  
warnings.filterwarnings('ignore')

#Import Data set using the pandas read_csv Api.
titanic_train = pd.read_csv('../input/train.csv')
titanic_test = pd.read_csv('../input/test.csv')
print(titanic_train.head())
titanic_test.head()
# Lets Remove unwanted columns that might be a integral part for the prediction.
titanic_train = titanic_train.drop(titanic_train.columns[[3,8]], axis=1)# The name column is not really going to help in predicting, so droping it from the dataframe.
titanic_train = titanic_train.drop(titanic_train.columns[8], axis = 1) # Removing the cabin column as it contains more nan than data.

# Cleaning Test Data
titanic_test = titanic_test.drop(titanic_test.columns[[2,7]], axis=1)# The name column is not really going to help in predicting, so droping it from the dataframe.
titanic_test = titanic_test.drop(titanic_test.columns[7], axis = 1) # Removing the cabin column as it contains more nan than data.
# lets clean the data and convert strings to numerics, so that we can use it in our machine learning models.
print(titanic_test.count())
titanic = titanic_train.fillna(0) # droping na as Age has a lot of na's, will improvise in the next version by aproximately padding na's of Age. 
titanic_test = titanic_test.fillna(0)
print(titanic_test.count())
# Converting strings to numeric data,so that our model can understand the data.
titanic = titanic.replace('S',1).replace('C',2).replace('Q',3) # Converting Embarked feature to numeric data.
titanic = titanic.replace('male',1).replace('female',2) # Converting Sex feature to numeric data.

# Cleaning for test
titanic_test = titanic_test.replace('S',1).replace('C',2).replace('Q',3) # Converting Embarked feature to numeric data.
titanic_test = titanic_test.replace('male',1).replace('female',2) # Converting Sex feature to numeric data.
# Seperating the data into feature and class
titanic_train = titanic.drop(titanic_train.columns[1], axis=1)
titanic_class = titanic['Survived']
print(titanic_train.head())
print(titanic_class.head())
# lets use train and split classifier from sklearn

x_titanic_train,x_titanic_test,y_titanic_train,y_titanic_test = train_test_split(titanic_train,titanic_class,test_size = 0.3,random_state = 93)
#Lets test the Acuracy and Prediction with SVM
svmclf = SVC(kernel='poly', degree=1)
svmclf.fit(x_titanic_train, y_titanic_train) 
y_titanic_pred = svmclf.predict(x_titanic_test)
print(accuracy_score(y_titanic_test, y_titanic_pred))
print(f1_score(y_titanic_test, y_titanic_pred, average='weighted'))
# Lets try the Acuracy and Prediction score with Random Forest.
rf = RandomForestClassifier(n_estimators=100, max_depth=4,random_state=0) # My testing shows depth 4 can give almost the best, there is no point increasing depth just for 2-3%.
rf.fit(x_titanic_train, y_titanic_train) 
y_titanic_pred = rf.predict(x_titanic_test)
print(accuracy_score(y_titanic_test, y_titanic_pred))
print(f1_score(y_titanic_test, y_titanic_pred, average='weighted'))
# Lets try the Acuracy and Prediction score with Decision Tree.
dtclf = DecisionTreeClassifier(random_state=0)
dtclf.fit(x_titanic_train, y_titanic_train) 
y_titanic_pred = dtclf.predict(x_titanic_test)
print(accuracy_score(y_titanic_test, y_titanic_pred))
print(f1_score(y_titanic_test, y_titanic_pred, average='weighted'))
# Lets try the Acuracy and Prediction score with K-nearest neighbor.
kn = KNeighborsClassifier(n_neighbors=150)
kn.fit(x_titanic_train, y_titanic_train) 
y_titanic_pred = kn.predict(x_titanic_test)
print(accuracy_score(y_titanic_test, y_titanic_pred))
print(f1_score(y_titanic_test, y_titanic_pred, average='weighted'))
# Lets try the Acuracy and Prediction score with Neural Network.
clf = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',beta_1=0.9, beta_2=0.999, early_stopping=False,epsilon=1e-08, hidden_layer_sizes=(5000, 200),learning_rate='constant', learning_rate_init=0.001,max_iter=200, momentum=0.9,nesterovs_momentum=True, power_t=0.5, random_state=1,shuffle=True, solver='lbfgs', tol=0.0001,validation_fraction=0.1, verbose=False, warm_start=False)
clf.fit(x_titanic_train, y_titanic_train) 
y_titanic_pred = clf.predict(x_titanic_test)
print(accuracy_score(y_titanic_test, y_titanic_pred))
print(f1_score(y_titanic_test, y_titanic_pred, average='weighted'))

y_titanic_pred = clf.predict(titanic_test)
print(y_titanic_pred)
Submission = pd.DataFrame({'PassengerId': titanic_test.PassengerId, 'Survived': y_titanic_pred})
Submission.to_csv('Submission.csv', index=False)
Submission.head()
