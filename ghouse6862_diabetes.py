# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/diabetes.csv')
df.head()
df.info()
df.describe()
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set_style('whitegrid')
sns.boxplot(x = 'Outcome',y = 'Age',data = df)
sns.boxplot(x = 'Outcome',y = 'Pregnancies',data = df)
sns.jointplot(x='Age',y='Pregnancies',data=df,kind='scatter')
sns.lmplot(x='Age',y='Pregnancies',data=df,hue='Outcome',fit_reg=False)
from sklearn.model_selection import train_test_split
X = df.drop(labels='Outcome',axis=1)
y = df['Outcome']
X_train , X_test , y_train , y_test = train_test_split(X ,y , test_size = 0.2, random_state = 101)
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
parameters = {'gamma':[1.0,0.1,0.01,0.001,0.00001,0.000001,0.0000001],'C':[0.1,1,10,100,10000]}
grid_model = GridSearchCV(SVC(),param_grid=parameters)
grid_model.fit(X_train, y_train)
grid_model.best_params_
predictions = grid_model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
from sklearn.ensemble import RandomForestClassifier
parameter_grid = {'n_estimators':[50,55,60,65,75,80,85,90,95,100]}
grid_forest_model = GridSearchCV(RandomForestClassifier(),param_grid=parameter_grid)
grid_forest_model.fit(X_train,y_train)
grid_forest_model.best_params_
forest_pred = grid_forest_model.predict(X_test)
print(classification_report(y_test,forest_pred))
print(confusion_matrix(y_test,forest_pred))