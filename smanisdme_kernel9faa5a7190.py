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
import os
#os.chdir("F:\\Practice\\Machine Learning and Deep Learning\\Classes\\Assignment\\Kaggle\\2nd")
#df=pd.read_csv("breast-cancer-wisconsin-data.zip")

df = pd.read_csv("../input/data.csv")
df.shape
df.head()
df.columns
df.isnull().sum()
df['diagnosis'].isnull()
y=df['diagnosis']
y
x=df.loc[:,'radius_mean':'fractal_dimension_worst']
x
x.isnull().sum()
df = df.drop(["id", "Unnamed: 32"], axis=1)
df.shape
x.head()
y=y.map({'M':0,'B':1})
y.head()
scale=ss()
x=scale.fit_transform(x)
x.shape
y.shape
pca=PCA()
out=pca.fit_transform(x)
out.shape
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, shuffle = True )
dt = DecisionTreeClassifier()
rf = RandomForestClassifier(n_estimators=100)
etc = ExtraTreesClassifier(n_estimators=100)
gbm = GradientBoostingClassifier()
knc = KNeighborsClassifier()
xg = XGBClassifier()
#Train data
dt1 = dt.fit(x_train,y_train)
rf1 = rf.fit(x_train,y_train)

etc1 = etc.fit(x_train,y_train)

gbm1 = gbm.fit(x_train,y_train)

knc1 = knc.fit(x_train,y_train)

xg1 = xg.fit(x_train,y_train)
#predictions
y_pred_dt = dt1.predict(x_test)
y_pred_rf = rf1.predict(x_test)

y_pred_etc= etc1.predict(x_test)

y_pred_gbm= gbm1.predict(x_test)

y_pred_knc= knc1.predict(x_test)

y_pred_xg= xg1.predict(x_test)
#probability value
y_pred_dt_prob = dt1.predict_proba(x_test)
y_pred_rf_prob = rf1.predict_proba(x_test)
y_pred_etc_prob = etc1.predict_proba(x_test)

y_pred_gbm_prob= gbm1.predict_proba(x_test)

y_pred_knc_prob = knc1.predict_proba(x_test)

y_pred_xg_prob = xg1.predict_proba(x_test)
#accuracy
accuracy_score(y_test,y_pred_dt)
accuracy_score(y_test,y_pred_rf)
accuracy_score(y_test,y_pred_etc)
accuracy_score(y_test,y_pred_knc)
accuracy_score(y_test,y_pred_xg)
accuracy_score(y_test,y_pred_gbm)
#Confusion Matrix
confusion_matrix(y_test,y_pred_dt)
confusion_matrix(y_test,y_pred_rf)
confusion_matrix(y_test,y_pred_etc)
confusion_matrix(y_test,y_pred_gbm)
confusion_matrix(y_test,y_pred_knc)
confusion_matrix(y_test,y_pred_xg)
#ROC
fpr_dt, tpr_dt, thresholds = roc_curve(y_test, y_pred_dt_prob[: , 1], pos_label= 1)
fpr_rf, tpr_rf, thresholds = roc_curve(y_test, y_pred_rf_prob[: , 1], pos_label= 1)

fpr_etc, tpr_etc, thresholds = roc_curve(y_test, y_pred_etc_prob[: , 1], pos_label= 1)
fpr_knc, tpr_knc, thresholds = roc_curve(y_test, y_pred_gbm_prob[: , 1], pos_label= 1)

fpr_xg, tpr_xg, thresholds = roc_curve(y_test, y_pred_xg_prob[: , 1], pos_label= 1)

fpr_gbm, tpr_gbm,thresholds = roc_curve(y_test, y_pred_gbm_prob[: , 1], pos_label= 1)
#AUC values
auc(fpr_dt,tpr_dt)
auc(fpr_rf,tpr_rf)
auc(fpr_etc,tpr_etc)
auc(fpr_knc,tpr_knc)
auc(fpr_xg,tpr_xg)
auc(fpr_gbm,tpr_gbm)
#Precision/Recall/F-score
precision_recall_fscore_support(y_test,y_pred_dt)
precision_recall_fscore_support(y_test,y_pred_rf)
precision_recall_fscore_support(y_test,y_pred_etc)
precision_recall_fscore_support(y_test,y_pred_knc)
precision_recall_fscore_support(y_test,y_pred_xg)
precision_recall_fscore_support(y_test,y_pred_gbm)
# ROC curve 
fig = plt.figure(figsize=(14,10))

ax = fig.add_subplot(111)

ax.set_xlabel('False Positive Rate')

ax.set_ylabel('True Positive Rate')

ax.set_title('ROC curve for models')

ax.set_xlim([0.0, 1.0])

ax.set_ylim([0.0, 1.0])

ax.plot(fpr_dt, tpr_dt, label = "dt")

ax.plot(fpr_rf, tpr_rf, label = "rf")

ax.plot(fpr_etc, tpr_etc, label = "etc")

ax.plot(fpr_knc, tpr_knc, label = "knc")

ax.plot(fpr_xg, tpr_xg, label = "xg")

ax.plot(fpr_gbm, tpr_gbm, label = "gbm")

ax.legend(loc="lower right")

plt.show()