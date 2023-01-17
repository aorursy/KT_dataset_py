# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score

import xgboost


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Reading data
data = pd.read_csv("/kaggle/input/cardiovascular-disease-dataset/cardio_train.csv", sep=";")
data.head()
data.info()
# Know the dimension
data.shape
# Remove the id
data.drop("id",axis=1,inplace=True)
# To find abnormal data
data.describe()
# To have a closer look to ap_hi
plt.hist(data["ap_hi"], bins = 200)
plt.show()
# To have a closer look to ap_lo
plt.hist(data["ap_lo"], bins = 200)
plt.show()
# Remove outliers
data = data[data["ap_lo"] < 200]
data = data[data["ap_hi"] < 200]
data = data[data["ap_lo"] > 30]
data = data[data["ap_hi"] > 30]
# Check if the target is balanced
data["cardio"].value_counts()
# Data after removing outliers
data.describe()
# Feature engineering
data["bmi"] = data["weight"]/ (data["height"]/100)**2
# Data preprocessing - train test split and normalise
y = data["cardio"]
X = data.drop(["cardio"], axis = 1)
X = normalize(X)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=8017)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=8017)
# Training (baseline)
dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier()
knn = KNeighborsClassifier()
svc = SVC(random_state=1)
log = LogisticRegression(solver="liblinear", max_iter=200)

models = {"Decision tree" : dtc,
          "Random forest" : rfc,
          "KNN" : knn,
          "SVM" : svc,
          "Logistic" : log}
scores= { }

for key, value in models.items():    
    model = value
    accuracies = cross_val_score(estimator=value, X=X_train_val, y=y_train_val, cv=4)
    scores[key] = round(sum(accuracies)/len(accuracies), 4)
    print("done. run {}".format(key))
    

# print score
print(scores)
# tuning rfc
grid = {"n_estimators" : np.arange(70,200,20)}
rfc_grid = GridSearchCV(rfc, grid, cv=4) 
rfc_grid.fit(X_train_val,y_train_val)

print("Best n_estimators: {}".format(rfc_grid.best_params_)) 
print("Best score: {}".format(rfc_grid.best_score_))

# tuning rfc
grid = {"n_estimators" : np.arange(115,150,5)}
rfc_grid = GridSearchCV(rfc, grid, cv=4) 
rfc_grid.fit(X_train_val,y_train_val)

print("Best n_estimators: {}".format(rfc_grid.best_params_)) 
print("Best score: {}".format(rfc_grid.best_score_))
# tuning knn

grid = {"n_neighbors" : np.arange(2,40,2)}
knn_grid = GridSearchCV(knn, grid, cv=4)
knn_grid.fit(X_train_val,y_train_val)# Fit)

print("Best n_neighbors: {}".format(knn_grid.best_params_)) 
print("Best score: {}".format(knn_grid.best_score_))

# tuning knn

grid = {"n_neighbors" : np.arange(38,50,2)}
knn_grid = GridSearchCV(knn, grid, cv=4)
knn_grid.fit(X_train_val,y_train_val)# Fit)

print("Best n_neighbors: {}".format(knn_grid.best_params_)) 
print("Best score: {}".format(knn_grid.best_score_))

# tuning svm

grid = {"kernel" : ['linear', 'poly', 'rbf', 'sigmoid']}
svc_grid = GridSearchCV(svc, grid, cv=4)
svc_grid.fit(X_train_val,y_train_val)# Fit)

print("Best n_estimators: {}".format(svc_grid.best_params_)) 
print("Best score: {}".format(svc_grid.best_score_))

# tuning log

grid = {"penalty" : ["l1", "l2"],
         "C" : np.arange(60,80,2)} 
log_grid = GridSearchCV(log, grid, cv=4)
log_grid.fit(X_train_val, y_train_val)

# Print hyperparameter
print("Best grid: {}".format(log_grid.best_params_)) 
print("Best score: {}".format(log_grid.best_score_))
# Training (final)
rfc_final = RandomForestClassifier(n_estimators= 130)
knn_final = KNeighborsClassifier(n_neighbors= 38)
svc_final = SVC(random_state=1, kernel = 'linear')
log_final = LogisticRegression(solver="liblinear", max_iter=200, C = 64, penalty= 'l1')

models = {"Random forest" : rfc_final,
          "KNN" : knn_final,
          "SVM" : svc_final,
          "Logistic" : log_final}
scores_final = { }

for key, value in models.items():    
    model = value
    model.fit(X_train_val, y_train_val)
    y_pred = model.predict(X_test)
    scores_final[key] = accuracy_score(y_pred, y_test)
    print("done. run {}".format(key))
    
print(scores_final)