import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
%matplotlib inline
train_df=pd.read_csv('../input/amazon-employee-access-challenge/train.csv')
train_df.head()
target_df=pd.DataFrame(train_df['ACTION'],columns=['ACTION'])
target_df['ACTION'].value_counts()
sns.catplot('ACTION',data=target_df,kind='count')
plt.title('Action distributions',size=20)
train_df.dtypes
train_df.isna().any()
train_df.drop('ACTION',axis=1,inplace=True)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV,train_test_split,cross_val_score
from sklearn.metrics import confusion_matrix,roc_curve,precision_recall_curve,auc
from sklearn.preprocessing import StandardScaler
param_grid={'n_neighbors':[3,5,7]}
knn=KNeighborsClassifier()
X=train_df.values
y=target_df['ACTION'].values
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=0)
grid_search=GridSearchCV(knn,param_grid,scoring='roc_auc')
grid_result=grid_search.fit(X_train,y_train)
grid_result.best_params_
grid_result.score(X_train,y_train)
scores=[]
for i in range(1,13):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    cv_scores=cross_val_score(knn,X,y,cv=15)
    scores.append(cv_scores.mean())
    
neighbors=np.arange(1,13)
plt.figure(figsize=(10,8))
sns.set(style='white')
plt.plot(neighbors,scores,color='b')
plt.axvline(7,color='g')
plt.axhline(0.94,color='red')
plt.title('Accuracy Vs Neighbors',size=20)
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy score')
ticks=np.arange(1,13)
plt.xticks(ticks)
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,y_train)
y_knn_pred=knn.predict(X_test)
knn_score=knn.score(X_test,y_test)
knn_score
conf_mat_knn=confusion_matrix(y_test,y_knn_pred)
sns.heatmap(conf_mat_knn,annot=True,fmt='g',cmap='gnuplot')
plt.xlabel('Predicted')
plt.ylabel('Actual')
from sklearn.svm import SVC
svm=SVC(gamma=1e-07,C=1e9)
svm.fit(X_train,y_train)
svm.score(X_train,y_train)
y_pred_svc=svm.predict(X_test)
svm.score(X_test,y_test)
conf_mat_svc=confusion_matrix(y_test,y_pred_svc)
sns.heatmap(conf_mat_svc,annot=True,fmt='g',cmap='summer')
y_svc=svm.fit(X_train,y_train).decision_function(X_test)
fpr,tpr,_=roc_curve(y_test,y_svc)
plt.plot(fpr,tpr,color='indianred')
plt.plot([0,1],[0,1],color='blue',linestyle='--')
plt.xlabel('False postive rate')
plt.ylabel('True positive rate')
auc_svc=auc(fpr,tpr)
plt.title('ROC curve for SVC with AUC: {0:.2f}'.format(auc_svc))
precision,recall,threshold=precision_recall_curve(y_test,y_svc)
closest_zero=np.argmin(np.abs(threshold))
closest_zero_p=precision[closest_zero]
closest_zero_r = recall[closest_zero]
plt.plot(precision,recall)
plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)
plt.title('Precision-Recall curve with SVC')
plt.xlabel('Precision')
plt.ylabel('Recall')
from sklearn.linear_model import LogisticRegression
reg_log=LogisticRegression()
reg_log.fit(X_train,y_train)
reg_log.score(X_train,y_train)
y_pred_log=reg_log.predict(X_test)
reg_log.score(X_test,y_test)
conf_mat_log=confusion_matrix(y_pred_log,y_test)
sns.heatmap(conf_mat_log,annot=True,fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
y_log=reg_log.fit(X_train,y_train).decision_function(X_test)
fpr,tpr,_=roc_curve(y_test,y_log)
plt.plot(fpr,tpr,color='indianred')
plt.plot([0,1],[0,1],color='blue',linestyle='--')
plt.xlabel('False postive rate')
plt.ylabel('True positive rate')
auc_svc=auc(fpr,tpr)
plt.title('ROC curve for Logistic regression with AUC: {0:.2f}'.format(auc_svc))
from sklearn.linear_model import SGDClassifier
sgd=SGDClassifier()
sgd.fit(X_train,y_train)
sgd.score(X_train,y_train)
y_pred_sgd=sgd.predict(X_test)
sgd.score(X_test,y_test)
conf_mat_sgd=confusion_matrix(y_pred_sgd,y_test)
sns.heatmap(conf_mat_sgd,annot=True,fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
y_sgd=sgd.fit(X_train,y_train).decision_function(X_test)
fpr,tpr,_=roc_curve(y_test,y_sgd)
plt.plot(fpr,tpr,color='indianred')
plt.plot([0,1],[0,1],color='blue',linestyle='--')
plt.xlabel('False postive rate')
plt.ylabel('True positive rate')
auc_svc=auc(fpr,tpr)
plt.title('ROC curve for SGD with AUC: {0:.2f}'.format(auc_svc))
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)
dtc.score(X_train,y_train)
y_pred_dtc=dtc.predict(X_test)
dtc.score(X_test,y_test)
conf_mat_dtc=confusion_matrix(y_pred_dtc,y_test)
sns.heatmap(conf_mat_dtc,annot=True,fmt='g',cmap='winter')
plt.xlabel('Predicted')
plt.ylabel('Actual')
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
param_grid={'n_estimators':[3,5,7,9],'max_depth':[5,7,9]}
grid_search=GridSearchCV(rfc,param_grid,scoring='roc_auc')
grid_search.fit(X_train,y_train)
grid_search.best_params_
grid_search.score(X_train,y_train)
y_pred_rfc=grid_search.predict(X_test)
grid_search.score(X_test,y_test)
conf_mat_rfc=confusion_matrix(y_pred_rfc,y_test)
sns.heatmap(conf_mat_rfc,annot=True,fmt='g',cmap='gnuplot')
plt.xlabel('Predicted')
plt.ylabel('Actual')
from sklearn.ensemble import GradientBoostingClassifier
gbdt=GradientBoostingClassifier()
params={'max_depth':[6,7,8,10,12]}
grid_search=GridSearchCV(gbdt,params,scoring='roc_auc')
grid_search.fit(X_train,y_train)
print('Best parameter:{}'.format(grid_search.best_params_))
print('Best cross validated score: {:.2f}'.format(grid_search.best_score_))
grid_search.score(X_train,y_train)
y_pred_gbdt=grid_search.predict(X_test)
grid_search.score(X_test,y_test)
conf_mat_gbdt=confusion_matrix(y_pred_gbdt,y_test)
sns.heatmap(conf_mat_gbdt,annot=True,fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
import xgboost as xgb
xgb_class=xgb.XGBClassifier(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
xgb_class.fit(X_train,y_train)
xgb_class.score(X_train,y_train)
y_pred_xgb=xgb_class.predict(X_test)
xgb_class.score(X_test,y_test)
conf_mat_xgb=confusion_matrix(y_pred_xgb,y_test)
sns.heatmap(conf_mat_xgb,annot=True,fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
from lightgbm import LGBMClassifier
lgbm=LGBMClassifier(num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 30, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
lgbm.fit(X_train,y_train)
lgbm.score(X_train,y_train)
y_pred_lgbm=lgbm.predict(X_test)
lgbm.score(X_test,y_test)
conf_mat_lgbm=confusion_matrix(y_pred_lgbm,y_test)
sns.heatmap(conf_mat_lgbm,annot=True,fmt='g')
test_df=pd.read_csv('../input/amazon-employee-access-challenge/test.csv')
test_df.head()
test_id=pd.DataFrame(test_df.iloc[:,0],columns=['id'])
test_id.head()
test_df.drop('id',axis=1,inplace=True)
X_test=scaler.fit_transform(test_df)
X_test
y_final_knn=knn.predict(X_test)
knn_df=pd.DataFrame(columns=['Id','Action'])
knn_df['Action']=y_final_knn
knn_df['Id']=test_id['id']
knn_df.head()
y_final_dtc=dtc.predict(X_test)
dtc_df=pd.DataFrame(columns=['Id','Action'])
dtc_df['Action']=y_final_dtc
dtc_df['Id']=test_id['id']
dtc_df.head()
dtc_df.to_csv('DTC_predictions.csv',index=False)
