import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



creditdata = pd.read_csv("../input/creditcardfraud/creditcard.csv")
print(creditdata.head())
creditdata['Class'].value_counts()
creditdata['Class'].value_counts().plot(kind='bar')

plt.xlabel('Class')

plt.ylabel('Freq')
plt.hist(creditdata['Amount'],bins='auto',histtype='bar')
plt.hist(creditdata['Time'],bins='auto',histtype='bar')
from sklearn.preprocessing import scale

creditdata['Scaled_Amount']=scale(creditdata[['Amount']])
creditdata['Scaled_Time']=scale(creditdata[['Time']])
print(creditdata.head())
creditdata=creditdata.drop(['Amount','Time'],axis=1)
print(len(creditdata[creditdata.Class==1]))
no_of_frauds=492
no_of_normal=len(creditdata[creditdata.Class==1])
fraud_indices=creditdata[creditdata.Class==1].index
print(fraud_indices)
creditdata.Class.unique()
normal_indices=creditdata[creditdata.Class==0].index
print(normal_indices)
undersam_normal_indices = np.random.choice(normal_indices, no_of_frauds, replace = False)
undersam_indices=np.concatenate([fraud_indices,undersam_normal_indices])
newdata=creditdata.loc[undersam_indices,:]
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,precision_score
x = creditdata.ix[:,creditdata.columns!='Class']

y = creditdata.ix[:,creditdata.columns=='Class']

x_undersample = newdata.ix[:,newdata.columns!='Class']

y_undersample = newdata.ix[:,newdata.columns=='Class']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

x_under_train,x_under_test,y_under_train,y_under_test=train_test_split(x_undersample,y_undersample,test_size=0.3,random_state=0)
x.head()
lr=LogisticRegression(C=0.01)
lr.fit(x_under_train,y_under_train.values.ravel())##ravel to convert shape of y to array
y_under_predict =lr.predict(x_under_test)

recall=recall_score(y_under_test.values,y_under_predict)
print(recall)
y_predict=lr.predict(x_test)

recall1=recall_score(y_test.values,y_predict)
print(recall1)
##from sklearn import cross_validation

##kf = cross_validation.KFold(len(undersam_indices),4)

##return kf
##for iteration, indices in enumerate(kf,start=1):

## lr.fit(x_undersample.iloc[indices[0],:],y_undersample.iloc[indices[0],:].values.ravel())

##y_predict1=lr.predict(x_undersample.iloc[indices[1],:].values)
##from sklearn.metrics import recall_score

##recallnew=recall_score(y_undersample.iloc[indices[1],:].values,y_predict1)
precision=precision_score(y_under_predict,y_under_test.values)

print(precision)
precision1=precision_score(y_predict,y_test.values)

print(precision1)
confusionmatrix = confusion_matrix(y_under_test,y_under_predict)
print(confusionmatrix)
y_under_pred_score = lr.fit(x_under_train,y_under_train.values.ravel()).decision_function(x_under_test.values)

fpr,tpr,thresholds= roc_curve(y_under_test.values.ravel(),y_under_pred_score)

auc_graph = auc(fpr,tpr)
plt.plot(fpr,tpr,label='AUC = %0.4f'% auc_graph)

plt.legend()

plt.title('ROC curve')

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.show()
##applying decision trees.
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_under_train,y_under_train.values.ravel())

y_under_predict_tree =lr.predict(x_under_test)

recall_tree=recall_score(y_under_test.values,y_under_predict_tree)
print(recall_tree)


dt.fit(x_train,y_train.values.ravel())

y_predict_tree =lr.predict(x_test)

recall_tree1=recall_score(y_test.values,y_predict_tree)

print(recall_tree1)
##from sklearn.model_selection import KFold

##kfold=KFold(5,random_state=None)

##print(kfold)

y_under_pred_score_tree = lr.fit(x_under_train,y_under_train.values.ravel()).decision_function(x_under_test.values)

fprt,tprt,thresholdst= roc_curve(y_under_test.values.ravel(),y_under_pred_score_tree)

auc_graph_tree = auc(fprt,tprt)

plt.plot(fprt,tprt,label='AUC = %0.4f'% auc_graph_tree )

plt.legend()

plt.title('ROC curve')

plt.xlabel('FPR')

plt.ylabel('TPR')



plt.show()
confusionmatrixt = confusion_matrix(y_under_test,y_under_predict_tree)

print(confusionmatrixt)
from imblearn.over_sampling import SMOTE
sm = SMOTE(sampling_strategy='auto', random_state=None) ##higher values of prop of majority giving lower recall

x_res, y_res = sm.fit_resample(x, y.values.ravel())
x_res_train,x_res_test,y_res_train,y_res_test=train_test_split(x_res,y_res,test_size=0.3,random_state=0)
lr.fit(x_res_train,y_res_train)
y_res_predict =lr.predict(x_res_test)

recall_res=recall_score(y_res_test,y_res_predict)

recall_res
y_predict_smote =lr.predict(x_test)

recall_res1=recall_score(y_test,y_predict_smote)

recall_res1
confusionmatrix_smote = confusion_matrix(y_res_test,y_res_predict)

print(confusionmatrix_smote)
y_res_pred_score = lr.fit(x_res_train,y_res_train).decision_function(x_res_test)

fpr_smote,tpr_smote,thresholds_smote= roc_curve(y_res_test,y_res_pred_score)

auc_graph_smote = auc(fpr_smote,tpr_smote)

plt.plot(fpr_smote,tpr_smote,label='AUC = %0.4f'% auc_graph_smote)

plt.legend()

plt.title('ROC curve')

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.show()