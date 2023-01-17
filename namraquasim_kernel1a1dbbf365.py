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
import pandas as pd
import numpy as nm
import seaborn as sns
import matplotlib.pyplot as plt 
data=pd.read_csv("../input/heart-disease-dataset/heart.csv")
data.head()
data.tail()
data.describe()
data.info()
sns.countplot(data=data, x='age',hue='target')
sns.countplot(data=data, x='sex',hue='target')
sns.countplot(data=data, x='cp',hue='target')
sns.countplot(data=data, x='trestbps',hue='target')
sns.countplot(data=data, x='chol',hue='target')
sns.countplot(data=data, x='fbs',hue='target')
sns.countplot(data=data, x='restecg',hue='target')
sns.countplot(data=data, x='thalach',hue='target')
sns.countplot(data=data, x='exang',hue='target')
sns.countplot(data=data, x='oldpeak',hue='target')
sns.countplot(data=data, x='slope',hue='target')
sns.countplot(data=data, x='ca',hue='target')
sns.countplot(data=data, x='thal',hue='target')
data_x=data[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang', 'oldpeak', 'slope', 'ca', 'thal']]
data_y=data['target']
data_x
from sklearn.model_selection import GridSearchCV , train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
scaler = StandardScaler()
scaled_x = scaler.fit_transform(data_x)
scaled_x_data = pd.DataFrame(scaled_x  , columns = data_x.columns)
scaled_x_data.head()
x_train , x_test , y_train , y_test = train_test_split(scaled_x_data , data_y , test_size = 0.2, random_state=0)
x_train.shape , x_test.shape , y_train.shape , y_test.shape
models = [RandomForestClassifier() , SVC() , KNeighborsClassifier()]
param_grid = [{
    "n_estimators" : [50 , 70 , 100 , 150 , 200]
} , {
    "kernel" : ["rbf" , "poly"],
    "C" : [0.1 , 0.3 , 1 , 1.3 , 2 , 5, 10]
} , {
    "n_neighbors" : [5 , 10 , 15 , 20]
}]
scores = []
for i in range(len(models)):
    if i == 0:
        m = "Random Forest"
    elif i == 1 :
        m = "SVC"
    elif i == 2 :
        m = "KNN"
        
    grid = GridSearchCV(models[i] , param_grid = param_grid[i] , cv = 5 , scoring = "accuracy")
    grid.fit(x_train , y_train)
    scores.append({"model" : m , "best parameters" : grid.best_params_ , "best score" : grid.best_score_})

    
scores = pd.DataFrame(scores)
scores    
   
model= RandomForestClassifier(n_estimators = 100)
model.fit(x_train , y_train)
model.score(x_test , y_test)
yp = model.predict(x_test)
cm = confusion_matrix(yp , y_test)
sns.heatmap(cm , annot = True)