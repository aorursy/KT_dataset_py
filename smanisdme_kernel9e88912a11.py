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
#os.chdir("F:\\Practice\\Machine Learning and Deep Learning\\Classes\\Assignment\\Kaggle\\4th")
#os.listdir()

#ccf=pd.read_csv("creditcardfraud.zip")

ccf = pd.read_csv("../input/creditcard.csv")
ccf.head(3)
ccf.info()
ccf.describe()
ccf.shape
ccf.columns.values
ccf.dtypes.value_counts()
plt.style.use('ggplot')



f, ax = plt.subplots(figsize=(11, 15))



ax.set_facecolor('#fafafa')

ax.set(xlim=(-5, 5))

plt.ylabel('Variables')

plt.title("Overview Data Set")

ax = sns.boxplot(data = ccf.drop(columns=['Amount', 'Class', 'Time']), 

  orient = 'h', 

  palette = 'Set2')
f, (ax1, ax2) = plt.subplots(1,2,figsize =( 18, 8))

corr = ccf.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

sns.heatmap((ccf.loc[ccf['Class'] ==1]).corr(), vmax = .8, square=True, ax = ax1, cmap = 'afmhot', mask=mask);

ax1.set_title('Fraud')

sns.heatmap((ccf.loc[ccf['Class'] ==0]).corr(), vmax = .8, square=True, ax = ax2, cmap = 'YlGnBu', mask=mask);

ax2.set_title('Normal')

plt.show()
sns.countplot(x='Class', data=ccf)
(ccf.isnull()).apply(sum, axis = 0)
y = ccf.iloc[:,30]

X = ccf.iloc[:,0:30]
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
sm = SMOTE(random_state=42)

X_res, y_res = sm.fit_sample(X_train, y_train)
X_res.shape
y_res.shape
xg_res = xg.fit(X_res, y_res)
y_pred_xg_res = xg_res.predict(X_test)

y_pred_xg_res
y_pred_xg_res_prob = xg_res.predict_proba(X_test)

y_pred_xg_res_prob
print ('Accuracy using XGB and SMOTE',accuracy_score(y_test,y_pred_xg_res))
confusion_matrix(y_test,y_pred_xg_res)
fpr_xg_res, tpr_xg_res, thresholds = roc_curve(y_test,

                                 y_pred_xg_res_prob[: , 1],

                                 pos_label= 1

                                 )
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