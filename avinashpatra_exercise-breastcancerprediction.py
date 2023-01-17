import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import os

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost.sklearn import XGBClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import auc, roc_curve

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_recall_fscore_support

from sklearn.preprocessing import StandardScaler as ss

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

from sklearn import preprocessing
# os.chdir("E:/A ML/Exercise 4")
data = pd.read_csv("../input/data.csv")
data.head()
data.describe()
data.shape
col = data.columns
data['Unnamed: 32'].isnull().sum()
data = data.drop(['Unnamed: 32','id'],axis = 1)
data.head()
X = data.loc[:, "radius_mean" : "fractal_dimension_worst"]
y = data['diagnosis']
X.head()
y.head()
y=y.map({'M': 1,'B': 0})
y.head()
scale = ss()

X = scale.fit_transform(X)
X
pca = PCA()

out = pca.fit_transform(X)

out.shape 
pca.explained_variance_ratio_  
X_train, X_test, y_train, y_test = train_test_split(X,

                                                    y,

                                                    test_size = 0.3,

                                                    shuffle = True

                                                    )
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
accuracy_score(y_test,y_pred_dt)
accuracy_score(y_test,y_pred_rf)
accuracy_score(y_test,y_pred_etc)
accuracy_score(y_test,y_pred_knc)
accuracy_score(y_test,y_pred_xg)
accuracy_score(y_test,y_pred_gbm)
confusion_matrix(y_test,y_pred_dt)
confusion_matrix(y_test,y_pred_rf)
precision_recall_fscore_support(y_test,y_pred_dt)
fpr_dt, tpr_dt, thresholds = roc_curve(y_test, y_pred_dt_prob[: , 1], pos_label= 1)
auc(fpr_dt,tpr_dt)
fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111)

ax.set_xlabel('False Positive Rate')

ax.set_ylabel('True Positive Rate')

ax.set_title('ROC Curve')

ax.plot(fpr_dt, tpr_dt, label = "dt")

plt.show()
fpr_rf, tpr_rf, thresholds = roc_curve(y_test, y_pred_rf_prob[: , 1], pos_label= 1)

auc(fpr_rf,tpr_rf)
precision_recall_fscore_support(y_test,y_pred_rf)
ax.plot(fpr_rf, tpr_rf, label = "rf")

plt.show()
fpr_etc, tpr_etc, thresholds = roc_curve(y_test, y_pred_etc_prob[: , 1], pos_label= 1)

fpr_knc, tpr_knc, thresholds = roc_curve(y_test, y_pred_knc_prob[: , 1], pos_label= 1)

fpr_xg, tpr_xg, thresholds = roc_curve(y_test, y_pred_xg_prob[: , 1], pos_label= 1)

fpr_gbm, tpr_gbm, thresholds = roc_curve(y_test, y_pred_gbm_prob[: , 1], pos_label= 1)
auc(fpr_etc,tpr_etc)
auc(fpr_knc,tpr_knc)
auc(fpr_xg,tpr_xg)
auc(fpr_gbm,tpr_gbm)
fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111)

ax.set_xlabel('False Positive Rate')

ax.set_ylabel('True Positive Rate')

ax.set_title('ROC Curve')

ax.plot(fpr_dt, tpr_dt, label = "dt")

ax.plot(fpr_rf, tpr_rf, label = "rf")

ax.plot(fpr_etc, tpr_etc, label = "etc")

ax.plot(fpr_knc, tpr_knc, label = "knc")

ax.plot(fpr_xg, tpr_xg, label = "xg")

ax.plot(fpr_gbm, tpr_gbm, label = "gbm")

plt.show()