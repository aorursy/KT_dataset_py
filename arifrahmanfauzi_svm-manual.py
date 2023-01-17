import pandas as pd
import numpy as np
import seaborn as sns
sns.set()

from audioClassification import feature_extraction

from jupyterthemes import jtplot
jtplot.style(theme='grade3', grid=False, fscale=1.3)
features, labels = feature_extraction("dataset/data/")
print(features.shape)
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(features,labels,test_size=0.2, random_state=42) # 80% training 20% testing
from sklearn.svm import SVC
clf = SVC(kernel='rbf',C=21, decision_function_shape='ovr', gamma=0.00009, verbose=False, probability=True)
model = clf.fit(X_train,y_train)
from sklearn.externals import joblib
import pickle
# with open('svm_model','wb') as f:
#     pickle.dump(model,f)
y_pred = clf.predict(X_test)
clf.score(X_test,y_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_confusion_matrix
plot_confusion_matrix(clf,X_test,y_test, cmap='Blues')
print(classification_report(y_test,y_pred))
from scikitplot.metrics import plot_roc, plot_precision_recall
y_proba = clf.predict_proba(X_test)
plot_roc(y_test,y_proba, figsize=(15,5))
plot_precision_recall(y_test,y_proba, figsize=(13,5))
from scikitplot.estimators import plot_learning_curve
plot_learning_curve(clf,X_train,y_train, figsize=(5,3))
from sklearn.model_selection import learning_curve,ShuffleSplit

