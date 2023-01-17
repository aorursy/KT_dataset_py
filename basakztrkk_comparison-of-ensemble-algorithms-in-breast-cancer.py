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
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# warning library

import warnings
warnings.filterwarnings('ignore')

# data import

data = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")
data.drop(['Unnamed: 32','id'], inplace = True, axis = 1)

data.diagnosis = [ 1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1)

#standardization

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))

# %% Train test split

n_estimators = 10  #agac sayısı
random_state = 42

test_size = 0.3  # %30 test  %70 train
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = test_size, random_state = 42)

# %% KNN

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, Y_train)

y_pred_test_knn = knn.predict(X_test)
y_pred_train_knn = knn.predict(X_train)
    
cm_test_knn = confusion_matrix(Y_test, y_pred_test_knn)
cm_train_knn = confusion_matrix(Y_train, y_pred_train_knn)
    
acc_test_knn = accuracy_score(Y_test, y_pred_test_knn)
acc_train_knn = accuracy_score(Y_train, y_pred_train_knn)

print("KNN Test Score: {}, KNN Train Score: {}".format(acc_test_knn, acc_train_knn))

print("KNN CM Test: ",cm_test_knn)
print("KNN CM Train: ",cm_train_knn)

print()
# %% SVC

svc = SVC()
svc.fit(X_train, Y_train)


y_pred_test_svc = svc.predict(X_test)
y_pred_train_svc = svc.predict(X_train)
    
cm_test_svc = confusion_matrix(Y_test, y_pred_test_svc)
cm_train_svc = confusion_matrix(Y_train, y_pred_train_svc)
    
acc_test_svc = accuracy_score(Y_test, y_pred_test_svc)
acc_train_svc = accuracy_score(Y_train, y_pred_train_svc)

print("SVC Test Score: {}, SVC Train Score: {}".format(acc_test_svc, acc_train_svc))
print("SVC CM Test: ",cm_test_svc)
print("SVC CM Train: ",cm_train_svc)

print()

# %% Desiicion Tree

dt = DecisionTreeClassifier(random_state = random_state, max_depth = 2)
dt.fit(X_train,Y_train)

y_pred_test_dt = dt.predict(X_test)
y_pred_train_dt = dt.predict(X_train)

cm_test_dt = confusion_matrix(Y_test, y_pred_test_dt)
cm_train_dt = confusion_matrix(Y_train, y_pred_train_dt)

acc_test_dt = accuracy_score(Y_test, y_pred_test_dt)
acc_train_dt = accuracy_score(Y_train, y_pred_train_dt)

print("Desicion Tree Test Score: {}, DT Train Score: {}".format(acc_test_dt, acc_train_dt))
print("Desicion Tree CM Test: ",cm_test_dt)
print("Desicion Tree CM Train: ",cm_train_dt)

print()
# %% Random Forest

rf = RandomForestClassifier(n_estimators = n_estimators, random_state = random_state, max_depth = 2)
rf.fit(X_train, Y_train)

y_pred_test_rf = rf.predict(X_test)
y_pred_train_rf = rf.predict(X_train)

cm_test_rf = confusion_matrix(Y_test, y_pred_test_rf)
cm_train_rf = confusion_matrix(Y_train, y_pred_train_rf)

acc_test_rf = accuracy_score(Y_test, y_pred_test_rf)
acc_train_rf = accuracy_score(Y_train, y_pred_train_rf)


print("Random Forest Test Score: {}, Random Forest Train Score: {}".format(acc_test_rf, acc_train_rf))
print("Random Forest CM Test: ",cm_test_rf)
print("Random Forest CM Train: ",cm_train_rf)

print()
# %% AdaBoost 

ada = AdaBoostClassifier(base_estimator = dt, n_estimators = n_estimators, random_state = random_state)
ada.fit(X_train, Y_train)

y_pred_test_ada = ada.predict(X_test)
y_pred_train_ada = ada.predict(X_train)

cm_test_ada = confusion_matrix(Y_test, y_pred_test_ada)
cm_train_ada = confusion_matrix(Y_train, y_pred_train_ada)


acc_test_ada = accuracy_score(Y_test, y_pred_test_ada)
acc_train_ada = accuracy_score(Y_train, y_pred_train_ada)


print("AdaBoost Test Score: {}, AdaBoost Train Score: {}".format(acc_test_rf, acc_train_rf))
print("AdaBoost CM Test: ",cm_test_rf)
print("AdaBoost CM Train: ",cm_train_rf)

print()
# %%

comp = [acc_train_knn, acc_train_svc, acc_train_dt, acc_train_rf, acc_train_ada]
model_names = ["KNN","SVC","DT","R","AdaBoost"]

fig, ax  = plt.subplots()
ax.bar(model_names,comp,color = ("g","b","r","c","royalblue"),ecolor ="black", alpha = 0.5)
ax.invert_yaxis
ax.set_xlabel('Oranlar (Yüzdesel)')
ax.set_title('Modellerin Doğruluk Oranları')