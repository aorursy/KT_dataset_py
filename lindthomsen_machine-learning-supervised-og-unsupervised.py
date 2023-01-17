# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import cross_val_score
heart_data = pd.read_csv('/kaggle/input/health-care-data-set-on-heart-attack-possibility/heart.csv')
heart_data.describe()
knn = KNeighborsClassifier(n_neighbors=5)
svm = SVC()
decision_tree = tree.DecisionTreeClassifier()
nn = MLPClassifier()

## crossvalidation
cv_knn=cross_val_score(knn,heart_data_predict,heart_data_target,cv=20)
cv_svm=cross_val_score(svm,heart_data_predict,heart_data_target)
cv_dt=cross_val_score(decision_tree,heart_data_predict,heart_data_target)
cv_nn=cross_val_score(nn,heart_data_predict,heart_data_target)

print("KNN = "+str(np.mean(cv_knn)))
print("SVM = "+str(np.mean(cv_svm)))
print("DT = "+str(np.mean(cv_dt)))
print("NN = "+str(np.mean(cv_nn)))
