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
data = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

data.head(10)
data['quality'] = np.where(data['quality']<7,0,1)
data.head(10)
X = data[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']].copy()

y = data['quality'].copy()

X.shape, y.shape
from sklearn.model_selection import train_test_split, GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

grid_values = {'C': [0.1,1,10,100,200,500]}

model = LogisticRegression()

grid_acc = GridSearchCV(model, param_grid = grid_values)

grid_acc.fit(X_train_scaled, y_train)



print('Best Parameter: ',grid_acc.best_params_)

print('Best SCORE: ',grid_acc.best_score_)
lr = LogisticRegression(C = 0.1).fit(X_train_scaled, y_train)

predict_lr = lr.predict(X_test_scaled)

acc_lr = accuracy_score(y_test, predict_lr)

acc_lr
from sklearn.svm import SVC

svc = SVC().fit(X_train_scaled, y_train)

predict_svc = svc.predict(X_test_scaled)

acc_svc = accuracy_score(y_test,predict_svc)

acc_svc
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state = 0).fit(X_train_scaled, y_train)

predict_rfc = rfc.predict(X_test_scaled)

acc_rfc = accuracy_score(y_test, predict_rfc)

acc_rfc
grid_values = {'n_estimators': [5, 10, 15, 20 , 25, 50]}

model = RandomForestClassifier(random_state = 0)

grid_acc = GridSearchCV(model, param_grid = grid_values)

grid_acc.fit(X_train_scaled, y_train)



print('Best Parameter: ',grid_acc.best_params_)

print('Best Score: ',grid_acc.best_score_)
rfc = RandomForestClassifier(n_estimators = 20, random_state = 0).fit(X_train_scaled, y_train)

predict_rfc = rfc.predict(X_test_scaled)

acc_rfc = accuracy_score(y_test, predict_rfc)

acc_rfc