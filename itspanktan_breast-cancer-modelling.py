import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost.sklearn import XGBClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from sklearn.metrics import auc, roc_curve

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler as ss
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
data = pd.read_csv("../input/data.csv")

print("Information of data columns and data type:")

print(data.info())
print("Quick Glance of the data: ")

print(data.head())
print("Data Information: ")

print(data.describe())
print("Shape of the Data: ")

print(data.shape)
print("Original Values in the diagnosis column and their count: ")

print(data['diagnosis'].value_counts())



lb = preprocessing.LabelBinarizer()

data['diagnosis'] = lb.fit_transform(data['diagnosis'])
print("Looking for the data categories: ")

print(lb.classes_)
print("Check the bindarized data: ")

print(data['diagnosis'].value_counts())
data = data.drop(["id", "Unnamed: 32"], axis=1)
X = data.drop("diagnosis", axis=1)

y = data["diagnosis"].values

print(X.shape)

print(y.shape)
scale = ss()

X = scale.fit_transform(X)

print(X.shape)

print(X[:5,:])
pca = PCA(n_components = 0.95)

X = pca.fit_transform(X)

print(X.shape)

print(X[:5,:])
print(pca.explained_variance_ratio_)

print(pca.explained_variance_ratio_.cumsum())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle = True )
dt = DecisionTreeClassifier()

rf = RandomForestClassifier(n_estimators=100)

etc = ExtraTreesClassifier(n_estimators=100)

knc = KNeighborsClassifier()

xg = XGBClassifier(learning_rate=0.5, reg_alpha= 5, reg_lambda= 0.1)

gbm = GradientBoostingClassifier()
dt1 = dt.fit(X_train,y_train)

rf1 = rf.fit(X_train,y_train)

etc1 = etc.fit(X_train,y_train)

knc1 = knc.fit(X_train,y_train)

xg1 = xg.fit(X_train,y_train)

gbm1 = gbm.fit(X_train,y_train)
y_pred_dt = dt1.predict(X_test)

y_pred_rf = rf1.predict(X_test)

y_pred_etc= etc1.predict(X_test)

y_pred_knc= knc1.predict(X_test)

y_pred_xg= xg1.predict(X_test)

y_pred_gbm= gbm1.predict(X_test)
y_pred_dt_prob = dt1.predict_proba(X_test)

y_pred_rf_prob = rf1.predict_proba(X_test)

y_pred_etc_prob = etc1.predict_proba(X_test)

y_pred_knc_prob = knc1.predict_proba(X_test)

y_pred_xg_prob = xg1.predict_proba(X_test)

y_pred_gbm_prob= gbm1.predict_proba(X_test)
print("DecisionTreeClassifier: {0}".format(accuracy_score(y_test,y_pred_dt)))

print("RandomForestClassifier: {0}".format(accuracy_score(y_test,y_pred_rf)))

print("ExtraTreesClassifier: {0}".format(accuracy_score(y_test,y_pred_etc)))

print("KNeighborsClassifier: {0}".format(accuracy_score(y_test,y_pred_knc)))

print("XGBClassifier: {0}".format(accuracy_score(y_test,y_pred_xg)))

print("GradientBoostingClassifier: {0}".format(accuracy_score(y_test,y_pred_gbm)))
print("DecisionTreeClassifier: ")

print(confusion_matrix(y_test,y_pred_dt))

print("RandomForestClassifier: ")

print(confusion_matrix(y_test,y_pred_rf))

print("ExtraTreesClassifier: ")

print(confusion_matrix(y_test,y_pred_etc))

print("GradientBoostingClassifier: ")

print(confusion_matrix(y_test,y_pred_gbm))

print("KNeighborsClassifier: ")

print(confusion_matrix(y_test,y_pred_knc))

print("XGBClassifier: ")

print(confusion_matrix(y_test,y_pred_xg))
fpr_dt, tpr_dt, thresholds = roc_curve(y_test, y_pred_dt_prob[: , 1], pos_label= 1)

fpr_rf, tpr_rf, thresholds = roc_curve(y_test, y_pred_rf_prob[: , 1], pos_label= 1)

fpr_etc, tpr_etc, thresholds = roc_curve(y_test, y_pred_rf_prob[: , 1], pos_label= 1)

fpr_knc, tpr_knc, thresholds = roc_curve(y_test, y_pred_rf_prob[: , 1], pos_label= 1)

fpr_xg, tpr_xg, thresholds = roc_curve(y_test, y_pred_xg_prob[: , 1], pos_label= 1)

fpr_gbm, tpr_gbm,thresholds = roc_curve(y_test, y_pred_gbm_prob[: , 1], pos_label= 1)
print("DecisionTreeClassifier: {0}".format(auc(fpr_dt,tpr_dt)))

print("RandomForestClassifier: {0}".format(auc(fpr_rf,tpr_rf)))

print("ExtraTreesClassifier: {0}".format(auc(fpr_etc,tpr_etc)))

print("GradientBoostingClassifier: {0}".format(auc(fpr_gbm,tpr_gbm)))

print("KNeighborsClassifier: {0}".format(auc(fpr_knc,tpr_knc)))

print("XGBClassifier: {0}".format(auc(fpr_xg,tpr_xg)))
print("DecisionTreeClassifier: ")

print(precision_recall_fscore_support(y_test,y_pred_dt))

print("RandomForestClassifier: ")

print(precision_recall_fscore_support(y_test,y_pred_rf))

print("ExtraTreesClassifier: ")

print(precision_recall_fscore_support(y_test,y_pred_etc))

print("GradientBoostingClassifier: ")

print(precision_recall_fscore_support(y_test,y_pred_gbm))

print("KNeighborsClassifier: ")

print(precision_recall_fscore_support(y_test,y_pred_knc))

print("XGBClassifier: ")

print(precision_recall_fscore_support(y_test,y_pred_xg))

# Plot ROC curve now

fig = plt.figure(figsize=(12,10))

ax = fig.add_subplot(111)



# Connect diagonals

ax.plot([0, 1], [0, 1], ls="--")   # Dashed diagonal line



# Labels etc

ax.set_xlabel('False Positive Rate')

ax.set_ylabel('True Positive Rate')

ax.set_title('ROC curve for models')



# Set graph limits

ax.set_xlim([0.0, 1.0])

ax.set_ylim([0.0, 1.0])



# Plot each graph now

ax.plot(fpr_dt, tpr_dt, label = "dt")

ax.plot(fpr_rf, tpr_rf, label = "rf")

ax.plot(fpr_etc, tpr_etc, label = "etc")

ax.plot(fpr_knc, tpr_knc, label = "knc")

ax.plot(fpr_xg, tpr_xg, label = "xg")

ax.plot(fpr_gbm, tpr_gbm, label = "gbm")



# Set legend and show plot

ax.legend(loc="lower right")

plt.show()