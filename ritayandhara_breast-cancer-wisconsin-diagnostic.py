# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
print(os.listdir("../input/"))
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
data.info()
data.describe()
data.isnull().sum()
data.drop(columns=['Unnamed: 32'],inplace=True)
data.isnull().sum()
data.sample(5)
sns.catplot(x='diagnosis',kind='count',data=data,height=7);
data['diagnosis'].replace(['M','B'],[1,0],inplace=True)
X = data.iloc[:,2:-1].values
y = data.iloc[:,1].values
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
from sklearn.neighbors import KNeighborsClassifier
accuracy=[]
for i in range(2,int(np.sqrt(X_train.shape[0])+10)):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    accuracy.append(accuracy_score(knn.predict(X_test),y_test))
    
plt.plot(range(2,int(np.sqrt(X_train.shape[0])+10)),accuracy)
plt.grid();
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
accuracy_score(y_pred,y_test)
param_dict={
    "weights" : ["uniform", "distance"],
    "algorithm" : ["auto", "ball_tree", "kd_tree", "brute"],
    "metric" : ["minkowski"],
    "p" : [1,2]
}
from sklearn.model_selection import GridSearchCV
grid_knn=GridSearchCV(knn,param_grid=param_dict, cv=10, scoring = 'recall', n_jobs=-1)
grid_knn.fit(X_train,y_train)
grid_knn.best_estimator_
grid_knn.best_params_
grid_knn.best_score_
knn = grid_knn.best_estimator_
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
accuracy_score(y_pred,y_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_pred,y_test)
from sklearn.metrics import classification_report
print(classification_report(y_pred,y_test))
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()
param_dict={
    "criterion" : ["gini","entropy"],
    "max_depth" : [1,2,3,4,5,6,7,8,9,None],
    "max_features" : ["auto","sqrt","log2",None],
    "min_samples_leaf" : [1,2,3,4,5,6,7,8,9,None],
    "min_samples_split" : [1,2,3,4,5,6,7,8,9,None]
}

grid_dt=GridSearchCV(clf,param_grid=param_dict, cv=10, scoring = 'recall', n_jobs=-1)
grid_dt.fit(X_train,y_train)
grid_dt.best_estimator_
grid_dt.best_params_
grid_dt.best_score_
dt = grid_dt.best_estimator_
dt.fit(X_train,y_train)
y_pred=dt.predict(X_test)
accuracy_score(y_pred,y_test)
confusion_matrix(y_pred,y_test)
print(classification_report(y_pred,y_test))