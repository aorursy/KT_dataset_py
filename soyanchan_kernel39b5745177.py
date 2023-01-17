# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd   # 이 문단은 기본적으로 자주 쓰는 요소들, 판다스 

import numpy as np     

import matplotlib as plt                               #시각화 라이브러리

import matplotlib.pyplot as plt                        #시각화 라이브러리

import seaborn as sns                                  #시각화 라이브러리

from sklearn.preprocessing import LabelEncoder # 문자 데이터를 숫자로 바까주는 그냥 인코딩

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split  # X,Y 트레인 테스트 분리

from tqdm import tqdm_notebook        

from sklearn.ensemble import VotingClassifier      

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import mean_squared_error   #MSE  #MSE에 root를 씌우는 것이 RMSE

from sklearn.metrics import accuracy_score

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import train_test_split

from sklearn.ensemble import BaggingClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier  

# X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2 , random_state=10)

from sklearn.ensemble import VotingClassifier

from tqdm import tqdm_notebook

from sklearn.model_selection import cross_val_score

from tqdm import tqdm_notebook

from xgboost import XGBClassifier



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/smhrd12/train.csv') 

test = pd.read_csv('../input/smhrd12/test.csv')
train['sex']=pd.get_dummies(train['sex'],prefix='Sex')

test['sex']=pd.get_dummies(test['sex'],prefix='Sex')



encoder = LabelEncoder()

train['workclass'] = encoder.fit_transform(train['workclass'])

test['workclass'] = encoder.transform(test['workclass'])



encoder = LabelEncoder()

train['education'] = encoder.fit_transform(train['education'])

test['education'] = encoder.transform(test['education'])





encoder = LabelEncoder()

train['marital-status'] = encoder.fit_transform(train['marital-status'])

test['marital-status'] = encoder.transform(test['marital-status'])





encoder = LabelEncoder()

train['occupation'] = encoder.fit_transform(train['occupation'])

test['occupation'] = encoder.transform(test['occupation'])



encoder = LabelEncoder()

train['relationship'] = encoder.fit_transform(train['relationship'])

test['relationship'] = encoder.transform(test['relationship'])





encoder = LabelEncoder()

train['race'] = encoder.fit_transform(train['race'])

test['race'] = encoder.transform(test['race'])



encoder = LabelEncoder()

train['native-country'] = encoder.fit_transform(train['native-country'])

test['native-country'] = encoder.transform(test['native-country'])



# train['race']=pd.get_dummies(train['race'],prefix='race')

# test['race']=pd.get_dummies(test['race'],prefix='race')
train['rel+gain'] = train['relationship']+train['capital-gain']

train['loss-gain'] = train['capital-loss']-train['capital-gain']

train['rel+gain+edu'] = train['relationship']+train['capital-gain']+train['education-num']

train['rel+gain+edu+age'] = train['relationship']+train['capital-gain']+train['education-num']

train['(rel+gain+edu+age)*rel'] = (train['relationship']+train['capital-gain']+train['education-num']+train['capital-loss'])*train['relationship']
# from sklearn import preprocessing   # 정규화



# x = train.values #returns a numpy array

# min_max_scaler = preprocessing.MinMaxScaler()

# x_scaled = min_max_scaler.fit_transform(x)

# df = pd.DataFrame(x_scaled)
# del train['race']

# del train['education']

# del train['sex']

del train['no']

df2 = train

Y = df2['income']           # 예측하고 평가하기 위해 답 분리

del df2['income']           # 문제만 남기고 답 삭제

X = df2.iloc[:,:]
del test['no']



test['rel+gain'] = test['relationship']+test['capital-gain']

test['loss-gain'] = test['capital-loss']-test['capital-gain']

test['rel+gain+edu'] = test['relationship']+test['capital-gain']+test['education-num']

test['rel+gain+edu+age'] = test['relationship']+test['capital-gain']+test['education-num']

test['(rel+gain+edu+age)*rel'] = (test['relationship']+test['capital-gain']+test['education-num']+test['capital-loss'])*test['relationship']
# 랜덤 포레스트 



from sklearn.ensemble import ExtraTreesClassifier



X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2 , random_state=40)



xtree = ExtraTreesClassifier(n_estimators=100, random_state=100)

xtree.fit(X_train, Y_train)

y_pred = xtree.predict(X_test)

xtree_score = cross_val_score(xtree,X_train,Y_train,cv=5).mean()

print('Accuracy =', accuracy_score(Y_test, y_pred))

print("I : {}, 교차 검증 점수  : {:.3f}".format(0,xtree_score))

print("I : {}, 훈련 세트 정확도: {:.3f}".format(0,xtree.score(X_train, Y_train)))

print("I : {}, 테스트 세트 정확도: {:.3f}".format(0,xtree.score(X_test, Y_test)))

xtree_scoreT = cross_val_score(xtree,X_test,Y_test,cv=5).mean()

print("테스트 교차검증" ,xtree_scoreT)
# 익스트림



from sklearn.ensemble import RandomForestClassifier



X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2 , random_state=40)



rf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)

rf.fit(X_train, Y_train)

y_pred = rf.predict(X_test)

rf_score = cross_val_score(rf,X_train,Y_train,cv=5).mean()

print('Accuracy =', accuracy_score(Y_test, y_pred))

print("I : {}, 교차 검증 점수  : {:.3f}".format(0,rf_score))

print("I : {}, 훈련 세트 정확도: {:.3f}".format(0,rf.score(X_train, Y_train)))

print("I : {}, 테스트 세트 정확도: {:.3f}".format(0,rf.score(X_test, Y_test)))

rf_scoreT = cross_val_score(rf,X_test,Y_test,cv=5).mean()

print("테스트 교차검증" ,rf_scoreT)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2 , random_state=40)



rf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)

xtree = ExtraTreesClassifier(n_estimators=100, random_state=0)



voting = VotingClassifier(

    estimators=[('rf', rf), ('xtree', xtree)],

    voting='hard') #hard#soft(디포트)#Weighted

voting.fit(X_train, Y_train)

vo_score = cross_val_score(voting,X_train,Y_train,cv=7).mean()



print(",교차 검증 점수  : {:.3f}".format(vo_score))

print(",훈련 세트 정확도: {:.3f}".format(voting.score(X_train, Y_train)))

print(",테스트 세트 정확도: {:.3f}".format(voting.score(X_test, Y_test)))

print('================================================')





X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2 , random_state=40)







scaler = StandardScaler()

scaler.fit(X_train)

scaler.fit(test)

test_sc = scaler.transform(test)

test_sc = pd.DataFrame(test_sc,columns=test.columns)

X_train_sc = scaler.transform(X_train)

X_test_sc = scaler.transform(X_test)



X_train_sc = pd.DataFrame(X_train_sc,columns=X_train.columns)

X_test_sc = pd.DataFrame(X_test_sc,columns=X_train.columns)

gbrt = GradientBoostingClassifier(random_state=100,max_depth=5,learning_rate=0.12)   #0.12

gbrt.fit(X_train_sc, Y_train)

y_pred = gbrt.predict(X_test_sc)

gbrt_score = cross_val_score(gbrt,X_train_sc,Y_train,cv=5).mean()

print('Accuracy =', accuracy_score(Y_test, y_pred))

print("I : {}, 교차 검증 점수  : {:.3f}".format(0,gbrt_score))

print("I : {}, 훈련 세트 정확도: {:.3f}".format(0,gbrt.score(X_train_sc, Y_train)))

print("I : {}, 테스트 세트 정확도: {:.3f}".format(0,gbrt.score(X_test_sc, Y_test)))

gbrt_scoreT = cross_val_score(gbrt,X_test,Y_test,cv=5).mean()

print(gbrt_scoreT)

gbrt_pred = gbrt.predict(test_sc)



fi = gbrt.feature_importances_   # 중요한 특성 알아보기



importances = pd.DataFrame(fi,index=X.columns)



importances.sort_values(by=0,ascending=False)
for i in [150] :   #100,130,170

    ada = AdaBoostClassifier(       

    DecisionTreeClassifier(max_depth=3), n_estimators=200,  #200

    algorithm="SAMME.R", learning_rate=0.22)  #0.22

    ada.fit(X_train, Y_train)

    # DecisionTreeClassifier(max_depth=2), n_estimators=200, algorithm="SAMME.R", learning_rate=0.2)



    ada_score = cross_val_score(ada,X_train,Y_train,cv=10).mean()

    print("i : {} ,교차 검증 점수  : {:.3f}".format(i,ada_score))

    print("i : {} ,훈련 세트 정확도: {:.3f}".format(i,ada.score(X_train, Y_train)))

    print("i : {} ,테스트 세트 정확도: {:.3f}".format(i,ada.score(X_test, Y_test)))

    print('================================================')

    

# i : 100 ,교차 검증 점수  : 0.867

# i : 100 ,훈련 세트 정확도: 0.886

# i : 100 ,테스트 세트 정확도: 0.880

# ================================================

# i : 150 ,교차 검증 점수  : 0.866

# i : 150 ,훈련 세트 정확도: 0.886

# i : 150 ,테스트 세트 정확도: 0.880

# ================================================

# i : 200 ,교차 검증 점수  : 0.866

# i : 200 ,훈련 세트 정확도: 0.886

# i : 200 ,테스트 세트 정확도: 0.880

# ================================================

# i : 190 ,교차 검증 점수  : 0.866

# i : 190 ,훈련 세트 정확도: 0.886

# i : 190 ,테스트 세트 정확도: 0.880

# ================================================

# i : 210 ,교차 검증 점수  : 0.866

# i : 210 ,훈련 세트 정확도: 0.886

# i : 210 ,테스트 세트 정확도: 0.880

# ================================================

# i : 170 ,교차 검증 점수  : 0.867

# i : 170 ,훈련 세트 정확도: 0.886

# i : 170 ,테스트 세트 정확도: 0.880

# ================================================
xgb = XGBClassifier(objective ='reg:linear', colsample_bytree = 0.6, learning_rate = 0.35,

                    max_depth = 7,  n_estimators = 13)

xgb.fit(X_train, Y_train)

# DecisionTreeClassifier(max_depth=2), n_estimators=200, algorithm="SAMME.R", learning_rate=0.2)



xgb_score = cross_val_score(xgb,X_train,Y_train,cv=10).mean()

print("i : {} ,교차 검증 점수  : {:.3f}".format(0,xgb_score))

print("i : {} ,훈련 세트 정확도: {:.3f}".format(0,xgb.score(X_train, Y_train)))

print("i : {} ,테스트 세트 정확도: {:.3f}".format(0,xgb.score(X_test, Y_test)))

print('================================================')
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2 , random_state=40)



gbrt = GradientBoostingClassifier(random_state=101,max_depth=5,learning_rate=0.12)



ada = AdaBoostClassifier(       

DecisionTreeClassifier(max_depth=3), n_estimators=200, 

algorithm="SAMME.R", learning_rate=0.22)



xgb = XGBClassifier(objective ='reg:linear', colsample_bytree = 0.6, learning_rate = 0.35,

                    max_depth = 7,  n_estimators = 13)



voting_clf = VotingClassifier(

    estimators=[('ada', ada), ('GB', gbrt),('xgb',xgb)],

    voting='hard') #hard#soft(디포트)#Weighted

voting_clf.fit(X_train, Y_train)

vo_score = cross_val_score(voting_clf,X_train,Y_train,cv=7).mean()



print(",교차 검증 점수  : {:.3f}".format(vo_score))

print(",훈련 세트 정확도: {:.3f}".format(voting_clf.score(X_train, Y_train)))

print(",테스트 세트 정확도: {:.3f}".format(voting_clf.score(X_test, Y_test)))

print('================================================')



# Accuracy = 0.8848319399419894   randomstate = 101

# ,교차 검증 점수  : 0.868

# ,훈련 세트 정확도: 0.888

# ,테스트 세트 정확도: 0.885

# ================================================





# Accuracy = 0.8848319399419894  randomstate = 100

# ,교차 검증 점수  : 0.868

# ,훈련 세트 정확도: 0.888

# ,테스트 세트 정확도: 0.884

# ================================================

# scaler = StandardScaler()

# scaler.fit(X_train)

# X_train_sc = scaler.transform(X_train)

# X_test_sc = scaler.transform(X_test)



# X_train_sc = pd.DataFrame(X_train_sc,columns=X_train.columns)

# X_test_sc = pd.DataFrame(X_test_sc,columns=X_train.columns)



# fi = gbrt.feature_importances_   # 중요한 특성 알아보기



# importances = pd.DataFrame(fi,index=X.columns)



# importances.sort_values(by=0,ascending=False)
# bag2_clf = BaggingClassifier(

#     GradientBoostingClassifier(random_state=100,max_depth=5,learning_rate=0.12))#0.1  0.12>>0.879



# bag2_clf.fit(X_train, Y_train)

# y_pred = bag2_clf.predict(X_test)

# bag2_score = cross_val_score(bag2_clf,X_train,Y_train,cv=5).mean()

# y_pred = bag2_clf.predict(X_test)

# print('Accuracy =', accuracy_score(Y_test, y_pred))

# print("i {} ,교차 검증 점수  : {:.3f}".format(0,bag2_score))

# print("i {} ,훈련 세트 정확도: {:.3f}".format(0,bag2_clf.score(X_train, Y_train)))

# print("i {} ,테스트 세트 정확도: {:.3f}".format(0,bag2_clf.score(X_test, Y_test)))
# scaler = StandardScaler()

# scaler.fit(X_train)

# X_train_sc = scaler.transform(X_train)

# X_test_sc = scaler.transform(X_test)



# X_train_sc = pd.DataFrame(X_train_sc,columns=X_train.columns)

# X_test_sc = pd.DataFrame(X_test_sc,columns=X_train.columns)



# bag1_clf = BaggingClassifier(

#     GradientBoostingClassifier(random_state=100,max_depth=5,learning_rate=0.12))#0.1  0.12>>0.879



# bag1_clf.fit(X_train_sc, Y_train)

# y_pred = bag1_clf.predict(X_test_sc)

# bag1_score = cross_val_score(bag1_clf,X_train_sc,Y_train,cv=5).mean()



# print('Accuracy =', accuracy_score(Y_test, y_pred))

# print("i {} ,교차 검증 점수  : {:.3f}".format(0,bag1_score))

# print("i {} ,훈련 세트 정확도: {:.3f}".format(0,bag1_clf.score(X_train_sc, Y_train)))

# print("i {} ,테스트 세트 정확도: {:.3f}".format(0,bag1_clf.score(X_test_sc, Y_test)))
# pred2 = gbrt.predict(test)    # 결과 저장, 답안지 제출.



sub = pd.read_csv('../input/smhrd12/sample_submission.csv')

sub['income'] = gbrt_pred

sub.to_csv('submission.csv',index=False)