# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import time
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv',skiprows=range(1,149903891),nrows=3500000)
test=pd.read_csv('../input/test.csv')
#展示数据
train.columns=['ip','app','device','os','channel','click_time','attributed_timed','is_attributed']
print(train.head())
print('*'*10)
#print(test.head())
#数据的统计信息
print(train['is_attributed'].value_counts())
print(train[train['is_attributed']==1]['is_attributed'].sum()/len(train))
'''
0    1992862
1       7138
Name: is_attributed, dtype: int64
0.003569

调参时减小训练数据量
'''
y=train['is_attributed']
#'click_time','is_attributed','attributed_timed'
train.drop(['click_time','is_attributed','attributed_timed'],axis=1,inplace=True)#inplace=True代表更改原内存的值
#'click_id','click_time'
test.drop(['click_id','click_time'],axis=1,inplace=True)
def print_score(m,dt,y):
    print('Accuracy:[Train,Val]')
    res=[m.score(dt,y)]#验证得分
    if hasattr(m,'obb_score_'):res.append(m.obb_score_)#袋外验证得分
    print(res)
    
    print('Train Confusion Matrix')
    df_train_proba=m.predict_proba(dt)#预测概率，原始的值
    df_train_pre_indices=np.argmax(df_train_proba,axis=1)#找到每一行概率最大值的索引
    print(df_train_pre_indices)
    classes_train=np.unique(y)#类别个数
    preds_train=classes_train[df_train_pre_indices]
    print('*'*10)
    print(preds_train)
    skplt.plot_confusion_matrix(y,preds_train)
#提交数据集的模板
test_submission = pd.read_csv("../input/sample_submission.csv")
test_submission.head()
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation,metrics
from sklearn.grid_search import GridSearchCV
gbm4= GradientBoostingClassifier(learning_rate=0.1,n_estimators=100,max_depth=5, min_samples_leaf =90,
                                 min_samples_split =2,max_features=2, subsample=0.75, random_state=10)  
%time gbm4.fit(train,y)  
#y_pred= gbm4.predict(train)  
test_submission['is_attributed']= gbm4.predict_proba(test)[:,1]  
#print("Accuracy : %.4g" % metrics.accuracy_score(y.values, y_pred)  )
#print("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob)   )
'''
Accuracy : 0.9973
AUC Score (Train): 0.957531
'''
#保存结果
test_submission.to_csv('rf4_result.csv',index=False)
'''gbm0= GradientBoostingClassifier(random_state=10)  
%time gbm0.fit(train,y)  
#y_pred= gbm0.predict(X)  
#y_predprob= gbm0.predict_proba(X)[:,1]  #预测概率

GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.1, loss='deviance', max_depth=3,
              max_features=None, max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=100,
              presort='auto', random_state=10, subsample=1.0, verbose=0,
              warm_start=False)
'''
'''y_pred= gbm1.predict(train)  
y_predprob= gbm1.predict_proba(train)[:,1]  
print("Accuracy : %.4g" % metrics.accuracy_score(y.values, y_pred)  )
print("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob)  )

Accuracy : 0.9975
AUC Score (Train): 0.934203
'''
'''#调参来提高模型泛化能力
param_test1= {'n_estimators':[_ for _ in range(80,120,10)]}  
gsearch1= GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1,min_samples_split=300,min_samples_leaf=20,max_depth=8,
                     max_features='sqrt',subsample=0.8,random_state=10), param_grid= param_test1, scoring='roc_auc',iid=False,cv=5)  
gsearch1.fit(train,y)  
gsearch1.grid_scores_,gsearch1.best_params_, gsearch1.best_score_  

([mean: 0.93674, std: 0.01494, params: {'n_estimators': 80},
  mean: 0.93646, std: 0.01580, params: {'n_estimators': 90},
  mean: 0.93795, std: 0.01442, params: {'n_estimators': 100},
  mean: 0.93787, std: 0.01436, params: {'n_estimators': 110}],
 {'n_estimators': 100},
 0.9379545746007757)
'''
'''#对max_depth进行调参
param_test2= {'max_depth':[_ for _ in range(3,14,2)], 'min_samples_split':[_ for _ in range(2,8,2)]}  
gsearch2= GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1,n_estimators=100, min_samples_leaf=20, max_features='sqrt', subsample=0.8,random_state=10),  
param_grid= param_test2,  scoring='roc_auc',  iid=False,  cv=5)  
gsearch2.fit(train,y)  
gsearch2.grid_scores_,gsearch2.best_params_, gsearch2.best_score_  

([mean: 0.91991, std: 0.01615, params: {'max_depth': 3, 'min_samples_split': 2},
  mean: 0.91991, std: 0.01615, params: {'max_depth': 3, 'min_samples_split': 4},
  mean: 0.91991, std: 0.01615, params: {'max_depth': 3, 'min_samples_split': 6},
  mean: 0.93837, std: 0.01390, params: {'max_depth': 5, 'min_samples_split': 2},
  mean: 0.93837, std: 0.01390, params: {'max_depth': 5, 'min_samples_split': 4},
  mean: 0.93837, std: 0.01390, params: {'max_depth': 5, 'min_samples_split': 6},
  mean: 0.89703, std: 0.03756, params: {'max_depth': 7, 'min_samples_split': 2},
  mean: 0.89703, std: 0.03756, params: {'max_depth': 7, 'min_samples_split': 4},
  mean: 0.89703, std: 0.03756, params: {'max_depth': 7, 'min_samples_split': 6},
  mean: 0.87293, std: 0.02422, params: {'max_depth': 9, 'min_samples_split': 2},
  mean: 0.87293, std: 0.02422, params: {'max_depth': 9, 'min_samples_split': 4},
  mean: 0.87293, std: 0.02422, params: {'max_depth': 9, 'min_samples_split': 6},
  mean: 0.84115, std: 0.05386, params: {'max_depth': 11, 'min_samples_split': 2},
  mean: 0.84115, std: 0.05386, params: {'max_depth': 11, 'min_samples_split': 4},
  mean: 0.84115, std: 0.05386, params: {'max_depth': 11, 'min_samples_split': 6},
  mean: 0.88446, std: 0.04320, params: {'max_depth': 13, 'min_samples_split': 2},
  mean: 0.88446, std: 0.04320, params: {'max_depth': 13, 'min_samples_split': 4},
  mean: 0.88446, std: 0.04320, params: {'max_depth': 13, 'min_samples_split': 6}],
 {'max_depth': 5, 'min_samples_split': 2},
 0.9383689834014476)
'''
'''#min_samples_leaf一起调参。  
param_test3= {'min_samples_split':[_ for _ in range(2,8,2)],'min_samples_leaf':[_ for _ in range(60,101,10)]}  
gsearch3= GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1,n_estimators=100,max_depth=5,max_features='sqrt',subsample=0.8,random_state=10),  
param_grid= param_test3,  scoring='roc_auc',  iid=False,  cv=5,n_jobs=-1)  
gsearch3.fit(train,y)  
gsearch3.grid_scores_,gsearch3.best_params_, gsearch3.best_score_ 

([mean: 0.93795, std: 0.00871, params: {'min_samples_leaf': 60, 'min_samples_split': 2},
  mean: 0.93795, std: 0.00871, params: {'min_samples_leaf': 60, 'min_samples_split': 4},
  mean: 0.93795, std: 0.00871, params: {'min_samples_leaf': 60, 'min_samples_split': 6},
  mean: 0.93666, std: 0.01605, params: {'min_samples_leaf': 70, 'min_samples_split': 2},
  mean: 0.93666, std: 0.01605, params: {'min_samples_leaf': 70, 'min_samples_split': 4},
  mean: 0.93666, std: 0.01605, params: {'min_samples_leaf': 70, 'min_samples_split': 6},
  mean: 0.93552, std: 0.01466, params: {'min_samples_leaf': 80, 'min_samples_split': 2},
  mean: 0.93552, std: 0.01466, params: {'min_samples_leaf': 80, 'min_samples_split': 4},
  mean: 0.93552, std: 0.01466, params: {'min_samples_leaf': 80, 'min_samples_split': 6},
  mean: 0.93907, std: 0.01263, params: {'min_samples_leaf': 90, 'min_samples_split': 2},
  mean: 0.93907, std: 0.01263, params: {'min_samples_leaf': 90, 'min_samples_split': 4},
  mean: 0.93907, std: 0.01263, params: {'min_samples_leaf': 90, 'min_samples_split': 6},
  mean: 0.93874, std: 0.01164, params: {'min_samples_leaf': 100, 'min_samples_split': 2},
  mean: 0.93874, std: 0.01164, params: {'min_samples_leaf': 100, 'min_samples_split': 4},
  mean: 0.93874, std: 0.01164, params: {'min_samples_leaf': 100, 'min_samples_split': 6}],
 {'min_samples_leaf': 90, 'min_samples_split': 2},
 0.9390701404941414)
'''
'''#用调优参数估计预测
gbm1= GradientBoostingClassifier(learning_rate=0.1, n_estimators=100,max_depth=5,min_samples_leaf =90, min_samples_split =2, 
                                 max_features='sqrt',subsample=0.8, random_state=10)  
gbm1.fit(train,y)  
y_pred= gbm1.predict(train)  
y_predprob= gbm1.predict_proba(train)[:,1]  
print("Accuracy : %.4g" % metrics.accuracy_score(y.values, y_pred)  )
print("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob)   )

Accuracy : 0.9972
AUC Score (Train): 0.952916
'''
#train.head()
'''param_test4= {'max_features':[_ for _ in range(1,6,1)]}  
gsearch4= GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1,n_estimators=100,max_depth=5, min_samples_leaf =90, min_samples_split =2,subsample=0.8, random_state=10),  
param_grid= param_test4,  scoring='roc_auc',  iid=False,  cv=5,n_jobs=-1)  
gsearch4.fit(train,y)  
gsearch4.grid_scores_,gsearch4.best_params_, gsearch4.best_score_   

([mean: 0.93640, std: 0.01683, params: {'max_features': 1},
  mean: 0.93907, std: 0.01263, params: {'max_features': 2},
  mean: 0.93895, std: 0.01413, params: {'max_features': 3},
  mean: 0.88630, std: 0.06103, params: {'max_features': 4},
  mean: 0.91486, std: 0.02613, params: {'max_features': 5}],
 {'max_features': 2},
 0.9390701404941414)
'''
'''param_test5= {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}  
gsearch5= GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1,n_estimators=100,max_depth=5, min_samples_leaf =90, 
 min_samples_split =2,max_features=2, random_state=10),  param_grid= param_test5,  scoring='roc_auc',  iid=False,   cv=5,n_jobs=-1)  
gsearch5.fit(train,y)  
gsearch5.grid_scores_,gsearch5.best_params_, gsearch5.best_score_   

([mean: 0.93699, std: 0.00974, params: {'subsample': 0.6},
  mean: 0.93832, std: 0.01243, params: {'subsample': 0.7},
  mean: 0.93926, std: 0.01257, params: {'subsample': 0.75},
  mean: 0.93907, std: 0.01263, params: {'subsample': 0.8},
  mean: 0.93739, std: 0.01086, params: {'subsample': 0.85},
  mean: 0.92796, std: 0.01406, params: {'subsample': 0.9}],
 {'subsample': 0.75},
 0.9392626480478425)
'''
'''#这时我们可以减半步长，最大迭代次数加倍来增加我们模型的泛化能力。再次拟合我们的模型：
gbm2= GradientBoostingClassifier(learning_rate=0.05, n_estimators=200,max_depth=5,min_samples_leaf =90, min_samples_split =2,
                                 max_features=2, subsample=0.75,random_state=10)  
gbm2.fit(train,y)  
y_pred= gbm2.predict(train)  
y_predprob= gbm2.predict_proba(train)[:,1]  
print("Accuracy : %.4g" % metrics.accuracy_score(y.values, y_pred)  )
print("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob)   )

Accuracy : 0.9972
AUC Score (Train): 0.953702
'''

'''gbm3= GradientBoostingClassifier(learning_rate=0.01, n_estimators=1000,max_depth=5,min_samples_leaf =90,  
               min_samples_split =2,max_features=2, subsample=0.75, random_state=10)  
gbm3.fit(train,y)  
y_pred= gbm3.predict(train)  
y_predprob= gbm3.predict_proba(train)[:,1]  
print("Accuracy : %.4g" % metrics.accuracy_score(y.values, y_pred)  )
print("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob)   )

Accuracy : 0.9973
AUC Score (Train): 0.956455
'''