import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as ss
from sklearn.decomposition import PCA
import os
#os.chdir("C:\\Users\\amit\\Desktop\\ELTM")
#os.listdir()

cancer = pd.read_csv("../input/data.csv")
cancer.shape 
cancer.head
cancer.dtypes
cancer.isnull().values.any()
cancer.isnull().sum()
cancer.drop(['id','Unnamed: 32'],axis=1,inplace=True)
cancer.shape
y=cancer.pop('diagnosis')
y[:25]
X=cancer
X is cancer
X.shape
X.dtypes
y=y.map({'M':1,'B':0})
y[:25]
scale=ss()
X=scale.fit_transform(X)
pca=PCA()
TX=pca.fit_transform(X)
pca.explained_variance_ratio_
pca.explained_variance_ratio_.cumsum()
final_X=TX[:,:10]
X1=final_X
X1 is final_X
X1_train,X1_test,y_train,y_test=train_test_split(X1,y,test_size=0.2,shuffle=True)
X1_train.shape
X1_test.shape
y_test.shape
dt=DecisionTreeClassifier()
rf=RandomForestClassifier(n_estimators=100,min_samples_leaf=5)
xg=XGBClassifier(learning_rate=0.5,

                   reg_alpha= 5,

                   reg_lambda= 0.1)
gb=GradientBoostingClassifier()
et=ExtraTreesClassifier(n_estimators=100)
kn=KNeighborsClassifier()
dt1=dt.fit(X1_train,y_train)
rf1=rf.fit(X1_train,y_train)
xg1=xg.fit(X1_train,y_train)
gb1=gb.fit(X1_train,y_train)
et1=et.fit(X1_train,y_train)
kn1=kn.fit(X1_train,y_train)
y_pred_dt=dt1.predict(X1_test)
y_pred_rf=rf1.predict(X1_test)
y_pred_xg=xg1.predict(X1_test)
y_pred_gb=gb1.predict(X1_test)
y_pred_et=et1.predict(X1_test)
y_pred_kn=kn1.predict(X1_test)
y_pred_dt
y_pred_rf
y_pred_xg
y_pred_gb
y_pred_et
y_pred_kn
y_pred_dt_prob=dt1.predict_proba(X1_test)

y_pred_rf_prob=rf1.predict_proba(X1_test)

y_pred_xg_prob=xg1.predict_proba(X1_test)

y_pred_gb_prob=gb1.predict_proba(X1_test)

y_pred_et_prob=et1.predict_proba(X1_test)

y_pred_kn_prob=kn1.predict_proba(X1_test)
y_pred_dt_prob
y_pred_xg_prob
accuracy_score(y_test,y_pred_dt)

accuracy_score(y_test,y_pred_rf)
accuracy_score(y_test,y_pred_xg)
accuracy_score(y_test,y_pred_gb)
accuracy_score(y_test,y_pred_kn)
accuracy_score(y_test,y_pred_et)
confusion_matrix(y_test,y_pred_dt)
confusion_matrix(y_test,y_pred_rf)
confusion_matrix(y_test,y_pred_xg)
confusion_matrix(y_test,y_pred_et)
confusion_matrix(y_test,y_pred_kn)
confusion_matrix(y_test,y_pred_gb)
precision_recall_fscore_support(y_test,y_pred_dt)
precision_recall_fscore_support(y_test,y_pred_rf)
precision_recall_fscore_support(y_test,y_pred_xg)
precision_recall_fscore_support(y_test,y_pred_gb)
precision_recall_fscore_support(y_test,y_pred_et)
precision_recall_fscore_support(y_test,y_pred_kn)
fpr_dt, tpr_dt, thresholds = roc_curve(y_test,

                                 y_pred_dt_prob[: , 1],

                                 pos_label= 1

                                 )
fpr_rf, tpr_rf, thresholds = roc_curve(y_test,

                                 y_pred_rf_prob[: , 1],

                                 pos_label= 1

                                 )
fpr_xg, tpr_xg, thresholds = roc_curve(y_test,

                                 y_pred_xg_prob[: , 1],

                                 pos_label= 1

                                 )
fpr_gb, tpr_gb,thresholds = roc_curve(y_test,

                                 y_pred_gb_prob[: , 1],

                                 pos_label= 1

                                 )
fpr_et, tpr_et,thresholds = roc_curve(y_test,

                                 y_pred_et_prob[: , 1],

                                 pos_label= 1

                                 )
fpr_kn, tpr_kn,thresholds = roc_curve(y_test,

                                 y_pred_kn_prob[: , 1],

                                 pos_label= 1

                                 )
auc(fpr_dt,tpr_dt)
auc(fpr_rf,tpr_rf)
auc(fpr_gb,tpr_gb)
auc(fpr_xg,tpr_xg)
auc(fpr_et,tpr_et)
auc(fpr_kn,tpr_kn)
fig = plt.figure(figsize=(10,10))          

ax = fig.add_subplot(111)

ax.plot([0, 1], [0, 1], ls="--")

ax.set_xlabel('False Positive Rate')  

ax.set_ylabel('True Positive Rate')

ax.set_title('ROC curve for models')

ax.set_xlim([0.0, 1.0])

ax.set_ylim([0.0, 1.0])

ax.plot(fpr_dt, tpr_dt, label = "dt")

ax.plot(fpr_rf, tpr_rf, label = "rf")

ax.plot(fpr_xg, tpr_xg, label = "xg")

ax.plot(fpr_gb, tpr_gb, label = "gb")

ax.plot(fpr_kn, tpr_kn, label = "kn")

ax.plot(fpr_et, tpr_et, label = "et")

ax.legend(loc="lower right")

plt.show()