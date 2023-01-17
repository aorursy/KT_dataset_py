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
#reading the data
data=pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
data.head()

#age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thai columns are featurd array
#target column is our target variable
data.describe()
# checking null values
data.isnull().sum()
#no null values present
#creating feature matrix and target array
y=data["target"]
X=data.drop("target",axis=1)
X.head()
y.head()
#splitting the data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.3)
#normalising the data
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(X_train)
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression
#make model
model=LogisticRegression()
model.fit(X_train_scaled,y_train)
#making predictions
y_pred=model.predict(X_test_scaled)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)
#accuracy using confusion matrix
accuracy=(33+40)/(33+40+8+10)
accuracy
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train_scaled,y_train)


y_predict_knn=knn.predict(X_test_scaled)
y_predict_knn
confusion_matrix(y_predict_knn,y_test)
acc_knn=69/91
acc_knn
knn.score(X_test_scaled,y_test)
accuracy_knn=[]
for k in range(1,15):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled,y_train)
    accuracy_knn.append([k,knn.score(X_test_scaled,y_test)])
    print(k,knn.score(X_test_scaled,y_test))

    
accuracy_knn
from sklearn.tree import DecisionTreeClassifier

model=DecisionTreeClassifier()
model.fit(X_train_scaled,y_train)
y_pred=model.predict(X_test_scaled)
y_pred
confusion_matrix(y_test,y_pred)
66/94
model.score(X_test_scaled,y_test)
# decisiontreeclassifier uses various hyperparameters like max_depth,max_leaf_nodes,min_samples_leaf

#using grid searchcv to find optimal max_depth
from sklearn.model_selection import GridSearchCV
parameters={"max_depth":[1,2,3,4,5,6,7,8,9,10]}
model=DecisionTreeClassifier()

grid=GridSearchCV(model,parameters,cv=10)
grid.fit(X_train_scaled,y_train)
grid.best_params_
# max_depth is found to be 5 for best accuracy
grid.score(X_test_scaled,y_test)
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(n_estimators=100,max_depth=5,max_features=12,oob_score=True,verbose=1,random_state=50)
model_rf.fit(X_train_scaled,y_train)
y_pred_rf=model_rf.predict(X_test_scaled)

confusion_matrix(y_test,y_pred)
acc=75/91
acc
model_rf.score(X_test,y_test)
#using grid searchcv to find optimal max_depth
from sklearn.model_selection import GridSearchCV
hyperparameters = {'max_features':np.arange(1,12),'max_depth':np.arange(1,6)}


model_tune=GridSearchCV(model_rf,hyperparameters,cv=10)
model_tune.fit(X_train_scaled,y_train)


model_tune.best_params_
y_pred_test_cv=model_tune.predict(X_test_scaled)     # predictions using tuned model
confusion_matrix(y_test,y_pred_test_cv)
76/91  #acc with tuned model
