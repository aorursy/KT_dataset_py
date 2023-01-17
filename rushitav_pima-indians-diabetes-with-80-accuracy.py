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
data=pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')

data.head()
features=['Glucose','BloodPressure','BMI','DiabetesPedigreeFunction','Age','Pregnancies','SkinThickness','Insulin']

X=data[features]

y=data['Outcome']

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)



#scaling the data to a common scale

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

X_train_scaled=scaler.fit_transform(X_train)

X_test_scaled=scaler.transform(X_test)

#finding best parameter for Logistic Regression using grid search

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

import warnings

warnings.filterwarnings('ignore')

grid_values={'C':[0.1,1,10,50,100,200,500]}

model=LogisticRegression()

grid_acc=GridSearchCV(model,param_grid=grid_values)

grid_acc.fit(X_train_scaled,y_train)



print('Grid best parameter:',grid_acc.best_params_)

print('Grid best score for Logistic Regression:',grid_acc.best_score_)



#Applying C=10 to predict the test data

lr=LogisticRegression(C=10)

lr.fit(X_train_scaled,y_train)

pre=lr.predict(X_test_scaled)

accuracy_score(y_test,pre)
#finding best parameter for KNN classifier using grid search



from sklearn.neighbors import KNeighborsClassifier

grid_values={'n_neighbors':[1,3,5,10,15,20]}

model=KNeighborsClassifier()

grid_acc=GridSearchCV(model,param_grid=grid_values)

grid_acc.fit(X_train_scaled,y_train)



print('Grid best parameter:',grid_acc.best_params_)

print('Grid best score for KNN:',grid_acc.best_score_)



#Applying n_neighbors=15 to predict the test data

knn=KNeighborsClassifier(n_neighbors=15)

knn.fit(X_train_scaled,y_train)

pre=knn.predict(X_test_scaled)

accuracy_score(y_test,pre)
#finding best parameter for rbf Support Vector Machine using grid search



from sklearn.svm import SVC

grid_values={'gamma':[0.001,0.01,0.1,1,10,50,100]}

model=SVC(kernel='rbf')

grid_acc=GridSearchCV(model,param_grid=grid_values)

grid_acc.fit(X_train_scaled,y_train)

y_decision_acc = grid_acc.decision_function(X_test) 



print('Grid best parameter:',grid_acc.best_params_)

print('Grid best score for SVM:',grid_acc.best_score_)



#Applying gamma=0.1 to predict the test data

svm=SVC(kernel='rbf',gamma=0.1)

svm.fit(X_train_scaled,y_train)

pre=svm.predict(X_test_scaled)

accuracy_score(y_test,pre)
#finding best parameter for Random Forest Classifier using grid search



from sklearn.ensemble import RandomForestClassifier

grid_values={'n_estimators':[5,10,15,20,50,100,150]}

model=RandomForestClassifier(random_state=0)

grid_acc=GridSearchCV(model,param_grid=grid_values)

grid_acc.fit(X_train_scaled,y_train)



print('Grid best parameter:',grid_acc.best_params_)

print('Grid best score for Random Forest:',grid_acc.best_score_)
#Applying n_estimators=100 to predict the test data

clf=RandomForestClassifier(n_estimators=100,random_state=0)

clf.fit(X_train_scaled,y_train)

pre=clf.predict(X_test_scaled)

accuracy_score(y_test,pre)