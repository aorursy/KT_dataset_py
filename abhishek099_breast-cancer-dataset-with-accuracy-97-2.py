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
data = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
data.head(10)
data = data.drop(['id', 'Unnamed: 32'], axis = 1)
data['diagnosis'] = np.where(data['diagnosis'] == 'M',1,0) 
data
data.isnull().sum()

import seaborn as sns
import matplotlib.pyplot as plt
g = sns.FacetGrid(data, col = 'diagnosis')
g.map(plt.hist, 'radius_mean', bins = 20)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
y = data['diagnosis'].copy()
X = data.drop('diagnosis', axis = 1).copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression().fit(X_train_scaled, y_train)
predict = lr.predict(X_test_scaled)
acc_lr = accuracy_score(y_test, predict)
acc_lr
from sklearn.svm import SVC
svc = SVC().fit(X_train_scaled, y_train)
predict = svc.predict(X_test_scaled)
acc_svc = accuracy_score(y_test, predict)
acc_svc
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier().fit(X_train_scaled, y_train)
predict = knn.predict(X_test_scaled)
acc_knn = accuracy_score(y_test, predict)
acc_knn
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier().fit(X_train_scaled, y_train)
predict = rfc.predict(X_test_scaled)
acc_rfc = accuracy_score(y_test, predict)
acc_rfc
from sklearn.model_selection import GridSearchCV
grid_values ={ 'n_estimators': [5, 10, 15, 20 ,25]}
model = RandomForestClassifier()
grid_acc = GridSearchCV(model, param_grid = grid_values)
grid_acc.fit(X_train_scaled, y_train)

print('Best Parameter: ', grid_acc.best_params_)
print('Best Score: ', grid_acc.best_score_)
rfc = RandomForestClassifier(n_estimators = 10).fit(X_train_scaled, y_train)
predict = rfc.predict(X_test_scaled)
acc_rfc = accuracy_score(y_test, predict)
acc_rfc