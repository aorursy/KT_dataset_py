import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split

from sklearn.metrics import balanced_accuracy_score, auc,accuracy_score,confusion_matrix,hamming_loss

from sklearn.metrics import precision_score, recall_score, precision_recall_curve,fbeta_score,roc_curve,roc_auc_score

import seaborn as sns



from imblearn.ensemble import BalancedRandomForestClassifier
ds = make_classification(n_samples=100000, n_features=14, n_classes=2, 

                         weights = [0.98,0.02], random_state=2020)



y = ds[1]



len(y[y == 1])
X,y = ds



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2020)
brf = BalancedRandomForestClassifier(n_estimators=1000, random_state=0)



brf.fit(X_train, y_train) 
y_pred = brf.predict(X_test)
print(accuracy_score(y_test, y_pred))

print(roc_auc_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

print(hamming_loss(y_test, y_pred))

print(precision_score(y_test, y_pred))

print(recall_score(y_test, y_pred))

print(fbeta_score(y_test, y_pred, beta=1))