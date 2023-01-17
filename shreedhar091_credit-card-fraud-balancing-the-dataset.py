%reset -f

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import  OneHotEncoder as ohe

from sklearn.preprocessing import StandardScaler as ss

from sklearn.compose import ColumnTransformer as ct
from xgboost.sklearn import XGBClassifier

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

from sklearn.metrics import average_precision_score

import sklearn.metrics as metrics

from sklearn.metrics import accuracy_score

from sklearn.metrics import auc, roc_curve
os.chdir("../input")

cr = pd.read_csv("creditcard.csv")
cr.head()
cr.info()
cr.describe()
cr.shape
cr.columns.values
cr.dtypes.value_counts()
f, axes = plt.subplots(2, 2, figsize=(7, 7))

sns.boxplot(x="Class", y="Time", data=cr, ax=axes[0, 0])

sns.boxplot(x="Class", y="V1", data=cr, ax=axes[0, 1])

sns.boxplot(x="Class", y="V2", data=cr, ax=axes[1, 0])

sns.boxplot(x="Class", y="V3", data=cr, ax=axes[1, 1])

plt.show()
cr.corr()

sns.heatmap(cr.corr())
sns.countplot(x='Class', data=cr)
(cr.isnull()).apply(sum, axis = 0)
y = cr.iloc[:,30]

X = cr.iloc[:,0:30]
X.shape              
X.columns
y.head()
X_trans = ss().fit_transform(X)

X_trans.shape
X_train, X_test, y_train, y_test =   train_test_split(X_trans,

                                                      y,

                                                      test_size = 0.3,

                                                      stratify = y

                                                      )





X_train.shape        
xg = XGBClassifier(learning_rate=0.5,

                   reg_alpha= 5,

                   reg_lambda= 0.1

                   )
xg_imb = xg.fit(X_train,y_train)
y_pred_xg_imb = xg_imb.predict(X_test)
y_pred_xg_imb_prob = xg_imb.predict_proba(X_test)
accuracy_score(y_test,y_pred_xg_imb)
confusion_matrix(y_test,y_pred_xg_imb)
fpr_xg_imb, tpr_xg_imb, thresholds = roc_curve(y_test,

                                 y_pred_xg_imb_prob[: , 1],

                                 pos_label= 1

                                 )
p_xg_imb,r_xg_imb,f_xg_imb,_ = precision_recall_fscore_support(y_test,y_pred_xg_imb)
print(auc(fpr_xg_imb,tpr_xg_imb))
p_xg_imb,r_xg_imb,f_xg_imb,_ = precision_recall_fscore_support(y_test,y_pred_xg_imb)
p_xg_imb
r_xg_imb
f_xg_imb
fig = plt.figure(figsize=(12,10)) 

ax = fig.add_subplot(111)

ax.plot([0, 1], [0, 1], ls="--")   

ax.set_xlabel('False Positive Rate')  

ax.set_ylabel('True Positive Rate')

ax.set_title('ROC curve for XGB and Unbalanced data')

ax.set_xlim([0.0, 1.0])

ax.set_ylim([0.0, 1.0])

ax.plot(fpr_xg_imb, tpr_xg_imb, label = "xgb")

ax.legend(loc="lower right")

plt.show()
sm = SMOTE(random_state=42)

X_res, y_res = sm.fit_sample(X_train, y_train)
X_res.shape
y_res.shape
xg_res = xg.fit(X_res, y_res)
#Predict

y_pred_xg_res = xg_res.predict(X_test)

y_pred_xg_res
y_pred_xg_res_prob = xg_res.predict_proba(X_test)

y_pred_xg_res_prob
#Accuracy

print ('Accuracy using XGB and SMOTE',accuracy_score(y_test,y_pred_xg_res))
#Confusion Matrix

confusion_matrix(y_test,y_pred_xg_res)
fpr_xg_res, tpr_xg_res, thresholds = roc_curve(y_test,

                                 y_pred_xg_res_prob[: , 1],

                                 pos_label= 1

                                 )

#Precion, Recall and F1 Score

p_xg_res,r_xg_res,f_xg_res,_ = precision_recall_fscore_support(y_test,y_pred_xg_res)
p_xg_res,r_xg_res,f_xg_res
print ('AUC using XGB and SMOTE',auc(fpr_xg_res,tpr_xg_res))
fig = plt.figure(figsize=(12,10)) 

ax = fig.add_subplot(111)

ax.plot([0, 1], [0, 1], ls="--")   

ax.set_xlabel('False Positive Rate')  

ax.set_ylabel('True Positive Rate')

ax.set_title('ROC curve for XGB and SMOTE')

ax.set_xlim([0.0, 1.0])

ax.set_ylim([0.0, 1.0])

ax.plot(fpr_xg_res, tpr_xg_res, label = "xgb")

ax.legend(loc="lower right")

plt.show()
ad = ADASYN(random_state=42)

X_ada, y_ada = sm.fit_sample(X_train, y_train)
X_ada.shape
y_ada.shape
xg_ada = xg.fit(X_ada, y_ada)
y_pred_xg_ada = xg_ada.predict(X_test)

y_pred_xg_ada
y_pred_xg_ada_prob = xg_ada.predict_proba(X_test)

y_pred_xg_ada_prob
print ('Accuracy using XGB and ADASYN',accuracy_score(y_test,y_pred_xg_ada))
confusion_matrix(y_test,y_pred_xg_ada)
fpr_xg_ada, tpr_xg_ada, thresholds = roc_curve(y_test,

                                 y_pred_xg_ada_prob[: , 1],

                                 pos_label= 1

                                 )
p_xg_ada,r_xg_ada,f_xg_ada,_ = precision_recall_fscore_support(y_test,y_pred_xg_ada)
p_xg_ada,r_xg_ada,f_xg_ada
print ('AUC using XGB and ADASYN',auc(fpr_xg_ada,tpr_xg_ada))
fig = plt.figure(figsize=(12,10))        

ax = fig.add_subplot(111)

ax.plot([0, 1], [0, 1], ls="--")

ax.set_xlabel('False Positive Rate')

ax.set_ylabel('True Positive Rate')

ax.set_title('ROC curve for XGB and ADASYN')

ax.set_xlim([0.0, 1.0])

ax.set_ylim([0.0, 1.0])

ax.plot(fpr_xg_ada, tpr_xg_ada, label = "xgb")

ax.legend(loc="lower right")

plt.show()