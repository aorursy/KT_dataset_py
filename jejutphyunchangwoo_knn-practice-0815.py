import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
breast_cancer = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data", header=None)
breast_cancer.head()
breast_cancer.columns = ['id_number', 'Clump_Thickness', 'Unif_Cell_Size', 'Unif_Cell_Shape', \

                         'Marg_Adhesion', 'Single_Epith_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses', 'Class']
breast_cancer.head()
breast_cancer.info()
breast_cancer.Bare_Nuclei.replace("?", np.NAN, inplace=True)
breast_cancer.Bare_Nuclei.unique()
breast_cancer.isnull().values.sum()
breast_cancer.Bare_Nuclei.fillna(breast_cancer.Bare_Nuclei.value_counts().index[0], inplace=True)
breast_cancer.Bare_Nuclei.unique()
breast_cancer["Bare_Nuclei"].isnull().sum()
breast_cancer["cancer_ind"] = 0
breast_cancer.loc[breast_cancer["Class"] == 4, "cancer_ind"] = 1
X_df = breast_cancer.drop(["id_number", "Class", "cancer_ind"], axis=1)

y = breast_cancer.cancer_ind
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled_df = scaler.fit_transform(X_df)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.3, random_state=42)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
y_pred = knn.predict(X_test)
accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)
roc_auc_score(y_test, y_pred)
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(knn, {"n_neighbors": [1,2,3,4,5]}, \

                          n_jobs=-1, cv=7, scoring="roc_auc")
grid_search.fit(X_train, y_train)
grid_search.best_params_
knn_best = grid_search.best_estimator_
y_pred = knn_best.predict(X_test)
accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)
roc_auc_score(y_test, y_pred)