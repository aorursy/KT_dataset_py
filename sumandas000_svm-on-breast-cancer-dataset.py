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
data=pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
data.sample(5)
data['diagnosis']=data['diagnosis'].replace('M',1)
data['diagnosis']=data['diagnosis'].replace('B',0)
X=data.iloc[:,2:32].values
y=data['diagnosis'].values
from sklearn.preprocessing import StandardScaler
clf=StandardScaler()
X=clf.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy_score(y_pred,y_test)
model1=SVC(kernel='rbf')
model1.fit(X_train,y_train)
y_pred=model1.predict(X_test)
accuracy_score(y_pred,y_test)
model2=SVC(kernel='poly',degree=1)
model2.fit(X_train,y_train)
y_pred=model2.predict(X_test)
accuracy_score(y_pred,y_test)
model3=SVC(kernel='linear')
model3.fit(X_train,y_train)
y_pred=model3.predict(X_test)
accuracy_score(y_pred,y_test)
model4=SVC(kernel='sigmoid')
model4.fit(X_train,y_train)
y_pred=model4.predict(X_test)
accuracy_score(y_pred,y_test)
param_dist={
    "C": [0.1,1,10,50,100],
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "degree": [1,2,3,4]
}
from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(model,param_grid=param_dist,cv=10,n_jobs=-1)
grid.fit(X_train,y_train)
a=grid.best_estimator_
b=grid.best_score_
c=grid.best_params_
print("Best estimator =",a)
print("Best accuracy score =",b)
print("Best Parameters =",c)