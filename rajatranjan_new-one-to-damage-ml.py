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
train_updated_1['has_repair_started'].fillna(value=0,inplace=True)
train_updated_1['count_families'].fillna(value=1.0,inplace=True)
# train_updated_1['count_families'].mode()
x,y=train_updated_1.loc[:,train_updated_1.columns!='damage_grade'],train_updated_1.loc[:,'damage_grade']
from sklearn.preprocessing import StandardScaler
m=StandardScaler()
xscaled=m.fit_transform(x)
xscaled
x_scaled=pd.DataFrame(data=xscaled,columns=x.columns)
x_scaled.head()
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
x_train,x_test,y_train, y_test = train_test_split(x_scaled, y, test_size=0.33, random_state=101)
# rf=RandomForestClassifier(n_estimators=50)
print(x_train.shape)
# rf.fit(x_train,y_train)
# y_pred=rf.predict(x_test)
# print('f1score',f1_score(y_test,y_pred,average='weighted'))
# from sklearn.decomposition import PCA
# pca = PCA(50)
# pca.fit(x_train)
# x_train = pca.transform(x_train)
# x_test = pca.transform(x_test)
# pca.explained_variance_ratio_
rf=RandomForestClassifier(n_estimators=100,n_jobs=-1,bootstrap=True,verbose=1,criterion='entropy')
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
print('f1score',f1_score(y_test,y_pred,average='weighted'))

rf.feature_importances_.mean()
col=pd.DataFrame({'importance': rf.feature_importances_, 'feature': x.columns})
main_col=col.sort_values(by=['importance'], ascending=[False])[:50]['feature'].values

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
x1= x[list(main_col)]
print(x1.head())
from sklearn.preprocessing import StandardScaler
m1=StandardScaler()
xscaled1=m1.fit_transform(x1)
x_train,x_test,y_train, y_test = train_test_split(xscaled1, y, test_size=0.33, random_state=101)
rf1=RandomForestClassifier(n_estimators=100,n_jobs=-1,bootstrap=True,verbose=1,criterion='entropy')
rf1.fit(x_train,y_train)
y_pred1=rf1.predict(x_test)
print('f1score',f1_score(y_test,y_pred1,average='weighted'))
print('classification_report',classification_report(y_test,y_pred1))
print('confusion_matrix\n',confusion_matrix(y_test,y_pred1))
from sklearn.model_selection import *
# kf = KFold(n_splits=5, shuffle=True)
# def runRF(x_train, y_train,x_test, y_test):
#     model=RandomForestClassifier(bootstrap=True, n_estimators=100,n_jobs=-1,verbose=1)
#     model.fit(x_train, y_train)
#     y_pred_train=model.predict(x_test)
#     #mse=rmsle(np.exp(y_pred_train)-1,np.exp(y_test)-1)
#     fscore=f1_score(y_test,y_pred_train,average='weighted')
# #     y_pred_test=model.predict(test)
#     return y_pred_train,fscore

# pred_full_test_RF = []  
# f1score=[]
# print('start..')
# for dev_index, val_index in kf.split(x):
#     dev_X, val_X = x.loc[dev_index], x.loc[val_index]
#     dev_y, val_y = y.loc[dev_index], y.loc[val_index]
#     ypred_valid_RF,fscore=runRF(dev_X, dev_y, val_X, val_y)
#     print("fold_ RF _ok "+str(fscore))
#     f1score.append(fscore)
#     print('ypred_valid_RF',ypred_valid_RF)
#     pred_full_test_RF.append(ypred_valid_RF)
# #     pred_full_test_RF = pred_full_test_RF + ypred_valid_RF
# print('f1score',f1score)
# plt.figure(figsize=(8,6))
# plt.scatter(x_train[:,0],x_train[:,1],c=y,cmap='plasma')
# plt.xlabel('First principal component')
# plt.ylabel('Second Principal Component')
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# print('classification_report',classification_report(y_test,y_pred))
# print('confusion_matrix\n',confusion_matrix(y_test,y_pred))
# col = pd.DataFrame({'importance': rf.feature_importances_, 'feature': x.columns})
# main_col=col.sort_values(by=['importance'], ascending=[False])['feature'].values
# main_col
# from sklearn.svm import SVC
# sv=SVC()
# sv.fit(x_train,y_train)
# y_pred=sv.predict(x_test)
# print('f1score',f1_score(y_test,y_pred,average='weighted'))
# main_col

# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train, y_test = train_test_split(m.fit_transform(x[main_col]), y, test_size=0.33, random_state=101)
# rf1=RandomForestClassifier(n_estimators=50,max_depth=12)
# rf1.fit(x_train,y_train)
# y_pred=rf1.predict(x_test)
# print('f1score',f1_score(y_test,y_pred,average='weighted'))
# from sklearn.tree import DecisionTreeClassifier
# clf = DecisionTreeClassifier(random_state=0)
# clf.fit(x_train,y_train)
# y_pred=clf.predict(x_test)
# print('f1score',f1_score(y_test,y_pred,average='weighted'))
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.linear_model import LogisticRegression
# lr=LogisticRegression(verbose=2)
# y_pred=lr.fit(x_train,y_train)
# print('f1score',f1_score(y_test,y_pred,average='weighted'))
# print('classification_report',classification_report(y_test,y_pred))
# print('confusion_matrix\n',confusion_matrix(y_test,y_pred))
# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB()
# gnb.fit(x_train,y_train)
# print('f1score',f1_score(y_test,gnb.predict(x_test),average='weighted'))
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.ensemble import GradientBoostingClassifier
# gb=GradientBoostingClassifier(loss='exponential',verbose=1,learning_rate=0.01,max_features='auto')
# gb.fit(x_train,y_train)
# y_pred=gb.predict(x_test)
# print('f1score',f1_score(y_test,y_pred,average='weighted'))
# print('classification_report',classification_report(y_test,y_pred))
# print('confusion_matrix\n',confusion_matrix(y_test,y_pred))
# gb=GradientBoostingClassifier(verbose=1,learning_rate=0.01,max_features='auto')
# gb.fit(x_train,y_train)
# y_pred=gb.predict(x_test)
# print('f1score',f1_score(y_test,y_pred,average='weighted'))
# print('classification_report',classification_report(y_test,y_pred))
# print('confusion_matrix\n',confusion_matrix(y_test,y_pred))
test_updated=test.merge(build, on=["building_id",'district_id','vdcmun_id'], how = 'inner')
test_updated_1=pd.get_dummies(test_updated.drop('building_id',axis=1),drop_first=True)
# build_ownership_1['building_id']=
test_updated_1.insert(loc=1, column='building_id', value=test_updated['building_id'])
test_updated_1=test_updated_1.drop('building_id',axis=1)
test_updated_1['has_repair_started'].fillna(value=0,inplace=True)
test_updated_1.shape

test_updated_11=m.transform(test_updated_1)
test_updated_111=pd.DataFrame(data=test_updated_11,columns=test_updated_1.columns)
print(test_updated_111.shape)
print(x_scaled.shape)
# from sklearn.model_selection import *
# kf = KFold(n_splits=7, shuffle=True)
# list(kf.split(xscaled))[0]
x_scaled=x_scaled[list(main_col)]
test_updated_111=test_updated_111[list(main_col)]
test_updated_111.head()
x_scaled.shape
from sklearn.model_selection import *
spl=10
kf = KFold(n_splits=spl, shuffle=True)
def runRF(x_train, y_train,x_test, y_test,test):
    model=RandomForestClassifier(bootstrap=True, n_estimators=110,n_jobs=-1,verbose=1)
    model.fit(x_train, y_train)
    y_pred_train=model.predict(x_test)
    #mse=rmsle(np.exp(y_pred_train)-1,np.exp(y_test)-1)
    fscore=f1_score(y_test,y_pred_train,average='weighted')
    y_pred_test=model.predict(test)
    return y_pred_train,fscore,y_pred_test

pred_full_test_RF = []  
f1score=[]
print('start..')
for dev_index, val_index in kf.split(x_scaled):
    dev_X, val_X = x_scaled.loc[dev_index], x_scaled.loc[val_index]
    dev_y, val_y = y.loc[dev_index], y.loc[val_index]
    ypred_valid_RF,fscore,y_pred_test_m=runRF(dev_X, dev_y, val_X, val_y,test_updated_111)
    print("fold_ RF _ok "+str(fscore))
    f1score.append(fscore)
    print('ypred_valid_RF',ypred_valid_RF)
    pred_full_test_RF.append(y_pred_test_m)
#     pred_full_test_RF = pred_full_test_RF + ypred_valid_RF
print('f1score',f1score)
np.array(pred_full_test_RF).shape
totpred=pd.DataFrame(data=pred_full_test_RF)
totpred
pred_full_test_RF[0]
sub=pd.read_csv('../input/a490e594-6-dataset/Dataset/sample_submission.csv')

for i in range(spl):
    subt=sub
    subt['damage_grade']=pred_full_test_RF[i]
    subt.to_csv('ml6p_fold'+str(i)+'.csv',index=False)
    

# int(totpred[0].apply(lambda x:int(x.split(" ")[1])).mean())
dd=[]
for j in range(421175):
    dd.append('Grade '+str(int(totpred[j].apply(lambda x:int(x.split(" ")[1])).mean())))
len(dd)
sub['damage_grade']=dd
sub.to_csv('ml6p_foldavg1.csv',index=False)
rf2=RandomForestClassifier(n_estimators=100,n_jobs=-1,bootstrap=True,verbose=1,criterion='entropy')
rf2.fit(x_scaled[list(main_col)],y)
y_pred2=rf2.predict(m.transform(test_updated_11[list(main_col)]))
sub['damage_grade']=dd
sub.to_csv('ml6p_0.csv',index=False)