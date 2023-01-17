# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
a_categorical=pd.read_csv('../input/train(1)_processed_continous.csv')
b_categorical=pd.read_csv('../input/test(1)_processed_continous.csv')
a_categorical.head()
a_categorical.shape,b_categorical.shape
combine=[a_categorical,b_categorical]
for dataset in combine:
    dataset['Age']=dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
a_categorical.head()
combine = [a_categorical, b_categorical]

for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4
    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)
a_categorical
combine = [a_categorical, b_categorical]
for dataset in combine:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
    dataset['not_alone'] = dataset['not_alone'].astype(int)
a2=a_categorical.drop(['Survived','Parch','SibSp'],axis=1)
b1=b_categorical.drop(['PassengerId','Parch','SibSp'],axis=1)
a2.shape,b1.shape
combine=[a2,b1]
for dataset in combine:
    dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['relatives']+1)
    dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)
a2.head()
a2.shape,b1.shape
a_array=a2.iloc[:,0:9].values
b_array=b1.iloc[:,0:9].values
a_array
from sklearn.preprocessing import OneHotEncoder
one_array=OneHotEncoder()
col=[0,1,2,3,5,8]
one_array.fit(a_array[:,col])
a2_encoded=one_array.transform(a_array[:,col])
b1_encoded=one_array.transform(b_array[:,col])
a2_encoded.shape
b1_encoded.shape
X_train = a2_encoded
Y_train = a_categorical["Survived"]
X_test  = b1_encoded
# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X_train,Y_train,test_size=0.3,random_state=1,shuffle=True)
from sklearn.metrics import accuracy_score
# stochastic gradient descent (SGD) learning
sgd = linear_model.SGDClassifier(max_iter=5, tol=None)    #number of times trained in dataset(max_iter=5)
sgd.fit(x_train, y_train)                                   
Y_pred = sgd.predict(x_test)

acc_sgd = round(accuracy_score(y_test, Y_pred) * 100, 2)

print(round(acc_sgd,2,), "%")
sgd.score(x_test,y_test)   #dataset(X_train,Y_train) where parch and sibsp are taken continous
# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)                        

Y_prediction = random_forest.predict(x_test)


acc_random_forest = round(accuracy_score(y_test,Y_prediction) * 100, 2)
print(round(acc_random_forest,2,), "%")
random_forest.score(x_test,y_test)         
# Logistic Regression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)            

Y_pred = logreg.predict(x_test)

acc_log = round(accuracy_score(y_test, Y_pred) * 100, 2)
print(round(acc_log,2,), "%")
logreg.score(x_test,y_test)             
# KNN
knn = KNeighborsClassifier(n_neighbors = 3)       
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

acc_knn = round(accuracy_score(y_test,y_pred) * 100, 2)
print(round(acc_knn,2,), "%")
knn.score(x_test,y_test)         
# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(x_train.toarray(), y_train)
                                                   
y_pred = gaussian.predict(x_test.toarray())

acc_gaussian = round(accuracy_score(y_test,y_pred) * 100, 2)
print(round(acc_gaussian,2,), "%")
gaussian.score(x_test.toarray(),y_test)       
# Linear SVC
linear_svc = LinearSVC()                 
linear_svc.fit(x_train, y_train)

Y_pred = linear_svc.predict(x_test)

acc_linear_svc = round(accuracy_score(y_test,Y_pred) * 100, 2)
print(round(acc_linear_svc,2,), "%")
linear_svc.score(x_test,y_test)         
# Decision Tree
decision_tree = DecisionTreeClassifier()      
decision_tree.fit(x_train, y_train)

Y_pred = decision_tree.predict(x_test)

acc_decision_tree = round(accuracy_score(y_test,Y_pred) * 100, 2)
print(round(acc_decision_tree,2,), "%")
decision_tree.score(x_test,y_test)       
results = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes',  
              'Stochastic Gradient Decent',                 
              'Decision Tree'],
    'Score': [acc_linear_svc, acc_knn, acc_log,                
              acc_random_forest, acc_gaussian, 
              acc_sgd, acc_decision_tree]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(9)
