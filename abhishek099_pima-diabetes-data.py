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
data = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')

data.head(10)
X = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]

y = data['Outcome']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

X_train.shape, y_train.shape, X_test.shape, y_test.shape
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()



X_train_normalize = scaler.fit_transform(X_train)

X_test_normalize = scaler.transform(X_test)
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

import warnings

warnings.filterwarnings('ignore')

grid_values = {'C': [0.1,1,10, 50, 100,200,500]}

model = LogisticRegression()

grid_acc = GridSearchCV(model, param_grid = grid_values)

grid_acc.fit(X_train_normalize, y_train)

print('Best grid parameter: ', grid_acc.best_params_)

print('Best score: ', grid_acc.best_score_)
from sklearn.metrics import accuracy_score

lr = LogisticRegression(C = 200).fit(X_train_normalize, y_train)

predictions_lr = lr.predict(X_test_normalize)

acc_lr = accuracy_score(y_test, predictions_lr)

acc_lr
from sklearn.svm import LinearSVC, SVC

grid_values = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 200, 500]}

model = LinearSVC()

grid_acc_linear = GridSearchCV(model, param_grid = grid_values)

grid_acc_linear.fit(X_train_normalize, y_train)



print('Best parameter: ',grid_acc_linear.best_params_)

print('Best score: ', grid_acc_linear.best_score_)
linearsvc = LinearSVC(C = 10).fit(X_train_normalize, y_train)

predictions_linearsvc = linearsvc.predict(X_test_normalize)

acc_linearsvc = accuracy_score(y_test, predictions_linearsvc)

acc_linearsvc
model = SVC()

grid_acc_svc = GridSearchCV(model, param_grid = grid_values)

grid_acc_svc.fit(X_train_normalize, y_train)

print('Best Parameter: ',grid_acc_svc.best_params_)

print('Best Score: ', grid_acc_svc.best_score_)
svc = SVC(C = 1).fit(X_train_normalize, y_train)

predictions_svc = svc.predict(X_test_normalize)

acc_svc = accuracy_score(y_test, predictions_svc)

acc_svc
from sklearn.neighbors import KNeighborsClassifier

grid_values = {'n_neighbors': [1, 3, 5, 10, 15, 20]}

model = KNeighborsClassifier()

grid_acc_knn = GridSearchCV(model, param_grid = grid_values)

grid_acc_knn.fit(X_train_normalize, y_train)



print('Best Parameter: ', grid_acc_knn.best_params_)

print('Best Score: ', grid_acc.best_score_)
knn = KNeighborsClassifier(n_neighbors = 20).fit(X_train_normalize, y_train)

predictions_knn = knn.predict(X_test_normalize)

acc_knn = accuracy_score(y_test, predictions_knn)

acc_knn
from sklearn.ensemble import RandomForestClassifier

grid_values = {'n_estimators': [5, 10, 15,20 ,30 ,40 ,50]}

model = RandomForestClassifier()

grid_acc_rfc = GridSearchCV(model, param_grid = grid_values)

grid_acc_rfc.fit(X_train_normalize, y_train)



print('Best Parameter: ', grid_acc_rfc.best_params_)

print('Best Score: ', grid_acc_rfc.best_score_)

rfc = RandomForestClassifier(n_estimators = 50).fit(X_train_normalize, y_train)

predictions_rfc = rfc.predict(X_test_normalize)

acc_rfc = accuracy_score(y_test, predictions_rfc)

acc_rfc
Models = pd.DataFrame({

    'Model': ['LogisticRegression', 'LinearSVC', 'SVC', 'KNeighborsClassifier', 'RandomForestClassifier'],

    'Accuracy Score': [acc_lr, acc_linearsvc, acc_svc, acc_knn, acc_rfc]

})

Models.sort_values(by = 'Accuracy Score', ascending = False)