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
train_data=pd.read_csv('/kaggle/input/titanic/train.csv')
test_data=pd.read_csv('/kaggle/input/titanic/test.csv')
train_data.info()


train_data.head()
test_data.info()
train_data.describe()
train_data["Survived"].value_counts()
train_data.isnull().sum()
test_data.isnull().sum()
import seaborn as sns
import matplotlib.pyplot as plt
f,ax=plt.subplots(1,1)
f.set_size_inches(10,6)
plt.title("Train")
sns.countplot(x=train_data['Survived'], ax=ax)

plt.show()

f,ax=plt.subplots(1,1)
f.set_size_inches(5,3)
sns.countplot(train_data["Survived"],hue=train_data["Sex"], ax=ax)
   
    
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(train_data[["Age"]])
train_data["Age"]=imp.transform(train_data[["Age"]])
imp.fit(test_data[["Age"]])
test_data["Age"]=imp.transform(test_data[["Age"]])
imp.fit(test_data[["Fare"]])
test_data["Fare"]=imp.transform(test_data[["Fare"]])

print(train_data['Embarked'].value_counts())
f,ax=plt.subplots(1,1)
f.set_size_inches(5,3)
sns.countplot(train_data["Survived"],hue=train_data["Embarked"], ax=ax)

train_data["Embarked"].fillna("S",inplace=True)
train_data.corr()
features=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])
Y=train_data['Survived']

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
poly = PolynomialFeatures()
X=poly.fit_transform(X)
X_test=poly.fit_transform(X_test)
scaler=StandardScaler()

scaler.fit(X)
X=scaler.transform(X)
X_test=scaler.transform(X_test)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.model_selection import train_test_split
X_train, X_val,Y_train,Y_val=train_test_split(X,Y,test_size=0.2,random_state=5)

model = SVC() 
model.fit(X_train, Y_train) 
  
# print prediction results 
predictions = model.predict(X_val) 
print(classification_report(Y_val, predictions)) 
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}

grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
grid.fit(X_train, Y_train)
print(grid.best_params_)
print(grid.best_estimator_) 
grid_predictions=grid.predict(X_val)
grid_predictions=grid.predict(X_val)
print(classification_report(Y_val, grid_predictions)) 
param_grid2 = {'C': [5,7], 'gamma': [0.1,0.01],'kernel': ['poly'],'poly_degree':[7,8,9]}

grid2 = GridSearchCV(SVC(), param_grid2, refit = True, verbose = 3) 
grid2.fit(X_train,Y_train)
