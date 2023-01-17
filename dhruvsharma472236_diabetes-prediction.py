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
data.info()
data.columns
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline 
bool=data.isna()
sns.heatmap(bool,yticklabels=False)
plt.figure(figsize=(14,6))
sns.barplot(x='Age',y='BMI',data=data)
sns.jointplot(x='Age',y='BMI',data=data)
plt.figure(figsize=(14,6))
sns.barplot(x='Age',y='BloodPressure',data=data)
sns.violinplot(x='Outcome',y='BMI',data=data)
from sklearn.model_selection import train_test_split
X=data.drop('Outcome',axis=1)
y=data['Outcome']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)
X_train.shape
y_train.shape
X_test.shape
y_test.shape
from sklearn.linear_model import LogisticRegression
model_lr=LogisticRegression()
model_lr.fit(X_train,y_train)
pred_lr=model_lr.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,pred_lr))
print(confusion_matrix(y_test,pred_lr))
from sklearn.tree import DecisionTreeClassifier
model_tree=DecisionTreeClassifier()
model_tree.fit(X_train,y_train)
pred_tree=model_tree.predict(X_test)
print(classification_report(y_test,pred_tree))
print(confusion_matrix(y_test,pred_lr))
from sklearn.ensemble import RandomForestClassifier 
model_rf=RandomForestClassifier()
model_rf.fit(X_train,y_train)
pred_rf=model_rf.predict(X_test)
print(classification_report(y_test,pred_tree))
from sklearn.svm import SVC
model_sv=SVC()
model_sv.fit(X_train,y_train)
pred_sv=model_sv.predict(X_test)
print(classification_report(y_test,pred_sv))
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
grid.fit(X_train,y_train)
grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test)
print(classification_report(y_test,grid_predictions))
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(data.drop('Outcome',axis=1))
scaled_features = scaler.transform(data.drop('Outcome',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=data.columns[:-1])
df_feat.head()
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=23)
x_train, x_test, Y_train, Y_test = train_test_split(scaled_features,data['Outcome'],
                                                    test_size=0.30)
knn.fit(x_train,Y_train)
pred_knn=knn.predict(x_test)
print(classification_report(Y_test,pred_knn))
from xgboost import XGBClassifier
model_xg=XGBClassifier()
model_xg.fit(x_train,Y_train)
pred_xr=model_xg.predict(x_test)
print(classification_report(Y_test,pred_xr))
