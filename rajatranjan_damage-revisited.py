# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
print(os.listdir("../input/a490e594-6-dataset/Dataset"))
#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import copy
import seaborn as sns

# Any results you write to the current directory are saved as output.
#training and test imports
train=pd.read_csv('../input/a490e594-6-dataset/Dataset/train.csv')
test=pd.read_csv('../input/a490e594-6-dataset/Dataset/test.csv')
build_ownership=pd.read_csv('../input/a490e594-6-dataset/Dataset/Building_Ownership_Use.csv')
build_struct=pd.read_csv('../input/a490e594-6-dataset/Dataset/Building_Structure.csv')
build_ownership_1=pd.get_dummies(build_ownership.drop('building_id',axis=1),drop_first=True)
# build_ownership_1['building_id']=
build_ownership_1.insert(loc=0, column='building_id', value=build_ownership['building_id'])
build_struct_1=pd.get_dummies(build_struct.drop('building_id',axis=1),drop_first=True)
# build_ownership_1['building_id']=
build_struct_1.insert(loc=0, column='building_id', value=build_struct['building_id'])
build = build_ownership_1.merge(build_struct_1, on=["building_id",'district_id','ward_id','vdcmun_id'], how = 'inner')
train_updated=train.merge(build, on=["building_id",'district_id','vdcmun_id'], how = 'inner')
train_updated_1=pd.get_dummies(train_updated.drop(['building_id','damage_grade'],axis=1),drop_first=True)
# build_ownership_1['building_id']=
train_updated_1.insert(loc=1, column='building_id', value=train_updated['building_id'])
train_updated_1.insert(loc=2, column='damage_grade', value=train_updated['damage_grade'])
train_updated_1.damage_grade.value_counts()
train_updated_1=train_updated_1.drop('building_id',axis=1)
# plt.figure(figsize=(15,15))
# sns.heatmap(train_updated_1.corr())
train_updated_1['has_repair_started'].fillna(value=0,inplace=True)
train_updated_1['count_families'].fillna(value=1.0,inplace=True)
x,y=train_updated_1.loc[:,train_updated_1.columns!='damage_grade'],train_updated_1.loc[:,'damage_grade']
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
x_train,x_test,y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

# rf=RandomForestClassifier(n_estimators=200,n_jobs=-1,bootstrap=True,verbose=1,max_features=30)
# rf.fit(x_train,y_train)
# y_pred=rf.predict(x_test)
# print('f1score',f1_score(y_test,y_pred,average='weighted'))
# print('classification_report',classification_report(y_test,y_pred))
# print('confusion_matrix\n',confusion_matrix(y_test,y_pred))
# rf.feature_importances_.mean()
# col=pd.DataFrame({'importance': rf.feature_importances_, 'feature': x.columns})
# main_col=col.sort_values(by=['importance'], ascending=[False])[:55]['feature'].values
# col[col.importance>=col.importance.mean()]

# x1= x[list(main_col)]

# x_train,x_test,y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=101)
# rf1=RandomForestClassifier(n_estimators=200,n_jobs=-1,bootstrap=True,verbose=1,max_features=33)
# rf1.fit(x_train,y_train)
# y_pred1=rf1.predict(x_test)
# print('f1score',f1_score(y_test,y_pred1,average='weighted'))
# print('classification_report',classification_report(y_test,y_pred1))
# print('confusion_matrix\n',confusion_matrix(y_test,y_pred1))
# mxfetaurs=30 76.81
# 40 f1score 0.7667369628208258
# 60 features 30 max_fea  f1score 0.7679366728364176
# rf1=RandomForestClassifier(n_estimators=100,n_jobs=-1,bootstrap=True,verbose=1,max_features=30)
# rf1.fit(x_train,y_train)
# y_pred1=rf1.predict(x_test)
# print('f1score',f1_score(y_test,y_pred1,average='weighted'))
# print('classification_report',classification_report(y_test,y_pred1))
# print('confusion_matrix\n',confusion_matrix(y_test,y_pred1))
# rf1=RandomForestClassifier(n_estimators=200,n_jobs=-1,bootstrap=True,verbose=1,max_features=26)
# rf1.fit(x_train,y_train)
# y_pred1=rf1.predict(x_test)
# print('f1score',f1_score(y_test,y_pred1,average='weighted'))
# print('classification_report',classification_report(y_test,y_pred1))
# print('confusion_matrix\n',confusion_matrix(y_test,y_pred1))
# rf1=RandomForestClassifier(n_estimators=100,n_jobs=-1,bootstrap=True,verbose=1,max_features=31)
# rf1.fit(x_train,y_train)
# y_pred1=rf1.predict(x_test)
# print('f1score',f1_score(y_test,y_pred1,average='weighted'))
# print('classification_report',classification_report(y_test,y_pred1))
# print('confusion_matrix\n',confusion_matrix(y_test,y_pred1))
# rf1=RandomForestClassifier(n_estimators=300,n_jobs=-1,bootstrap=True,verbose=1,max_features=30)
# rf1.fit(x_train,y_train)
# y_pred1=rf1.predict(x_test)
# print('f1score',f1_score(y_test,y_pred1,average='weighted'))
# print('classification_report',classification_report(y_test,y_pred1))
# print('confusion_matrix\n',confusion_matrix(y_test,y_pred1))
# rf1=RandomForestClassifier(n_estimators=350,n_jobs=-1,bootstrap=True,verbose=1,max_features=30)
# rf1.fit(x_train,y_train)
# y_pred1=rf1.predict(x_test)
# print('f1score',f1_score(y_test,y_pred1,average='weighted'))
# print('classification_report',classification_report(y_test,y_pred1))
# print('confusion_matrix\n',confusion_matrix(y_test,y_pred1))
# rf1=RandomForestClassifier(n_estimators=400,n_jobs=-1,bootstrap=True,verbose=1,max_features=25)
# rf1.fit(x_train,y_train)
# y_pred1=rf1.predict(x_test)
# print('f1score',f1_score(y_test,y_pred1,average='weighted'))
# print('classification_report',classification_report(y_test,y_pred1))
# print('confusion_matrix\n',confusion_matrix(y_test,y_pred1))
# rf1=RandomForestClassifier(n_estimators=400,n_jobs=-1,bootstrap=True,verbose=1,max_features=27)
# rf1.fit(x_train,y_train)
# y_pred1=rf1.predict(x_test)
# print('f1score',f1_score(y_test,y_pred1,average='weighted'))
# print('classification_report',classification_report(y_test,y_pred1))
# print('confusion_matrix\n',confusion_matrix(y_test,y_pred1))
# rf1=RandomForestClassifier(n_estimators=500,n_jobs=-1,bootstrap=True,verbose=1,max_features=30)
# rf1.fit(x_train,y_train)
# y_pred1=rf1.predict(x_test)
# print('f1score',f1_score(y_test,y_pred1,average='weighted'))
# print('classification_report',classification_report(y_test,y_pred1))
# print('confusion_matrix\n',confusion_matrix(y_test,y_pred1))
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# rfc=RandomForestClassifier(random_state=1994,verbose=1,n_estimators=300)
# param_grid = { 
#     'min_samples_split':[0.75,2,3,4]
# }
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=101)
# # bootstrap=True, class_weight=None, criterion='gini',
# #             max_depth=2, max_features='auto', max_leaf_nodes=None,
# #             min_impurity_decrease=0.0, min_impurity_split=None,
# #             min_samples_leaf=1, min_samples_split=2,
# #             min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
# #             oob_score=False, random_state=0, verbose=0, warm_start=False
# CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid,verbose=2,cv=2)
# print('Fitting Gridsearch..')
# CV_rfc.fit(x_train, y_train)


# print('best params',CV_rfc.best_params_)
# print('Predicting..')
# y_cvpred=CV_rfc.predict(x_test)

# from sklearn.metrics import f1_score
# print('F1 score..')
# print(f1_score(y_test,y_cvpred,average='weighted'))
test_updated=test.merge(build, on=["building_id",'district_id','vdcmun_id'], how = 'inner')
test_updated_1=pd.get_dummies(test_updated.drop('building_id',axis=1),drop_first=True)
# build_ownership_1['building_id']=
test_updated_1.insert(loc=1, column='building_id', value=test_updated['building_id'])
test_updated_1=test_updated_1.drop('building_id',axis=1)
test_updated_1['has_repair_started'].fillna(value=0,inplace=True)
test_updated_1.shape
def rfmodel(x,y,test,n_estimators,max_features,s,depth,min_samples_split):
    rf1=RandomForestClassifier(n_estimators=n_estimators,n_jobs=-1,bootstrap=True,verbose=1,max_features=max_features,max_depth=depth,min_samples_split=min_samples_split)
    print('fitting')
    rf1.fit(x,y)
    print('predicting')
    y_pred1=rf1.predict(test)
    sub=pd.read_csv('../input/a490e594-6-dataset/Dataset/sample_submission.csv')
    sub.damage_grade=y_pred1
    sub.to_csv('newSub'+str(s)+'.csv',index=False)
    sub.head()
# rfmodel(x,y,test_updated_1,450,30,7,4000) #73.998
# rfmodel(x,y,test_updated_1,450,40,8,2000) #73.908
# rfmodel(x,y,test_updated_1,450,30,9,None) 0.74014
# rfmodel(x,y,test_updated_1,400,30,10,None) 0.73994
# rfmodel(x,y,test_updated_1,350,30,11,None) 0.74024
# rfmodel(x,y,test_updated_1,400,27,12,None) 0.74011
# rfmodel(x,y,test_updated_1,500,30,13,None,2) 0.73992
# rfmodel(x,y,test_updated_1,300,30,14,None,2) 0.73984
# rfmodel(x,y,test_updated_1,300,30,15,None,4) 0.74511
# rfmodel(x,y,test_updated_1,350,30,16,None,4) 0.74440
# rfmodel(x,y,test_updated_1,280,30,17,None,4) 0.74438
# rfmodel(x,y,test_updated_1,300,30,18,None,5) 0.74598
# rfmodel(x,y,test_updated_1,250,30,19,None,4) 0.74465
# rfmodel(x,y,test_updated_1,250,30,20,None,5) 0.74566
# rfmodel(x,y,test_updated_1,300,30,21,None,6) 0.74666
# rfmodel(x,y,test_updated_1,350,30,22,None,6) 0.74682
# rfmodel(x,y,test_updated_1,300,30,23,None,7) 0.74740
# rfmodel(x,y,test_updated_1,350,30,24,None,7) 0.74734
# rfmodel(x,y,test_updated_1,300,30,25,None,8) 0.74774
# rfmodel(x,y,test_updated_1,300,30,21,None,6) 0.74666
# rfmodel(x,y,test_updated_1,350,30,22,None,6) 0.74682
# rfmodel(x,y,test_updated_1,300,30,23,None,7) 0.74740
# rfmodel(x,y,test_updated_1,350,30,24,None,7) 0.74734
# rfmodel(x,y,test_updated_1,300,30,25,None,8) 0.74774
# rfmodel(x,y,test_updated_1,300,30,26,None,10) 0.74836
# rfmodel(x,y,test_updated_1,300,30,27,None,12) 0.74897
# rfmodel(x,y,test_updated_1,300,30,28,None,14) 0.74921
# rfmodel(x,y,test_updated_1,350,30,29,None,14) 0.74886
# rfmodel(x,y,test_updated_1,300,30,30,None,20) 0.74897
# rfmodel(x,y,test_updated_1,300,30,31,None,16) 0.74923
# rfmodel(x,y,test_updated_1,300,30,32,None,18) 0.74895
# rfmodel(x,y,test_updated_1,320,30,33,None,16) 0.74934
# rfmodel(x,y,test_updated_1,320,30,34,None,17) 0.74928 --not good
# rfmodel(x,y,test_updated_1,350,30,35,None,17) 0.74958 good -increase estimators
# rfmodel(x,y,test_updated_1,320,30,36,None,18) 0.74905
# rfmodel(x,y,test_updated_1,350,30,37,None,18) 0.74916 nope 17 best estiators
# rfmodel(x,y,test_updated_1,400,30,38,None,17) 0.74974 good inrease estimators
# rfmodel(x,y,test_updated_1,450,30,39,None,18) 0.74880
# rfmodel(x,y,test_updated_1,370,30,40,None,17) 0.74936
# rfmodel(x,y,test_updated_1,500,30,41,None,19) 0.74910
# rfmodel(x,y,test_updated_1,450,30,42,None,17) 0.74938
# rfmodel(x,y,test_updated_1,500,30,43,None,17) 0.74950
# rfmodel(x,y,test_updated_1,600,30,44,None,17) 0.74918
# rfmodel(x,y,test_updated_1,1000,30,45,None,17) 0.74895
# nope..just 400 estimators and maxdepth 17