# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#导入常用包
import time 
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics,cross_validation
import scikitplot.plotters as skplt
from sklearn.model_selection import StratifiedKFold
#import matplotlib.pylab as plt
from sklearn.grid_search import GridSearchCV
%matplotlib inline
#导入数据
train=pd.read_csv('../input/train.csv',skiprows=160000000,nrows=2000000)
test=pd.read_csv('../input/test.csv')
#处理时间特征
'''def dataPreProcessTime(df):
    df['click_time'] = pd.to_datetime(df['click_time']).dt.date
    df['click_time'] = df['click_time'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
    
    return df'''
#展示数据
train.columns=['ip','app','device','os','channel','click_time','attributed_timed','is_attributed']
print(train.head())
print('*'*10)
print(test.head())
#处理点击时间
#train = dataPreProcessTime(train)
#test = dataPreProcessTime(test)
print(train.head())
print('*'*10)
print(test.head())
#数据的统计信息
print(train['is_attributed'].value_counts())
print(train[train['is_attributed']==1]['is_attributed'].sum()/len(train))
#不管任何参数，都用默认的，拟合数据看情况
'''rf0=RandomForestClassifier(oob_score=True,random_state=10,n_jobs=-1)
rf0.fit(train,y)
print(rf0.oob_score_)#输出袋外准确率，泛化能力体现
y_predprob=rf0.predict_proba(train)[:,1]#预测数据
print('AUC Score (Train):%f'% metrics.roc_auc_score(y,y_predprob))
#0.99715725,AUC Score (Train):0.999454,可见袋外分数已经很(理解为袋外数据作为验证集时的准确率，也就是模型的泛化能力)
#而且AUC分数也很高（AUC是指从一堆样本中随机抽一个，抽到正样本的概率比抽到负样本的概率 大的可能性）。
#相对于GBDT的默认参数输出，RF的默认参数拟合效果对本例要好一些。  '''
#首先对n_estimators进行网格搜索
'''param_test1={'n_estimators':[_ for _ in range(10,71,10)]}
#print(range(10,71,10))
gsearch1=GridSearchCV(estimator=RandomForestClassifier(min_samples_split=100,min_samples_leaf=20,max_depth=8,max_features='sqrt',
                                                       random_state=10),param_grid=param_test1,scoring='roc_auc',cv=5,n_jobs=-1)
gsearch1.fit(train,y)
print(gsearch1.grid_scores_)
print('*'*10)
print(gsearch1.best_params_)
print('*'*10)
print(gsearch1.best_score_)'''
'''
[mean: 0.91909, std: 0.02009, params: {'n_estimators': 10}, 
 mean: 0.92283, std: 0.02591, params: {'n_estimators': 20}, 
 mean: 0.93013, std: 0.02157, params: {'n_estimators': 30}, 
 mean: 0.92795, std: 0.02175, params: {'n_estimators': 40}, 
 mean: 0.92913, std: 0.02274, params: {'n_estimators': 50}, 
 mean: 0.93138, std: 0.02290, params: {'n_estimators': 60}, 
 mean: 0.93186, std: 0.02317, params: {'n_estimators': 70}]
**********
{'n_estimators': 70}
**********
0.9318593718982462
'''
#上面我们得到最佳的若学习迭代次数，接着我们对决策树最大深度max_depth和内部节点再划分所需最小样本数min_samples_split进行网格搜索
'''param_test2={'max_depth':[_ for _ in range(5,10,2)],'min_samples_split':[_ for _ in range(100,201,50)]}
gsearch2=GridSearchCV(estimator=RandomForestClassifier(n_estimators=60,min_samples_leaf=20,max_features='sqrt',oob_score=True,
                                                      random_state=10),param_grid=param_test2,scoring='roc_auc',iid=False,cv=5,n_jobs=-1)
gsearch2.fit(train,y)
print(gsearch2.grid_scores_)
print('*'*10)
print(gsearch2.best_params_)
print('*'*10)
print(gsearch2.best_score_)'''
'''
[mean: 0.91430, std: 0.02620, params: {'max_depth': 5, 'min_samples_split': 100}, 
 mean: 0.91427, std: 0.02509, params: {'max_depth': 5, 'min_samples_split': 150}, 
 mean: 0.91513, std: 0.02596, params: {'max_depth': 5, 'min_samples_split': 200}, 
 mean: 0.92603, std: 0.02233, params: {'max_depth': 7, 'min_samples_split': 100}, 
 mean: 0.92727, std: 0.02562, params: {'max_depth': 7, 'min_samples_split': 150}, 
 mean: 0.92734, std: 0.02648, params: {'max_depth': 7, 'min_samples_split': 200}, 
 mean: 0.93592, std: 0.02747, params: {'max_depth': 9, 'min_samples_split': 100}, 
 mean: 0.92949, std: 0.02930, params: {'max_depth': 9, 'min_samples_split': 150}, 
 mean: 0.92710, std: 0.02971, params: {'max_depth': 9, 'min_samples_split': 200}]
**********
{'max_depth': 9, 'min_samples_split': 100}
**********
0.9359221905382125
'''
#已经取了三个最优参数，看看现在模型的袋外分数：  
#rf1= RandomForestClassifier(n_estimators= 60, max_depth=9, min_samples_split=100,  
#                                 min_samples_leaf=20,max_features='sqrt' ,oob_score=True,random_state=10,n_jobs=-1)  
#rf1.fit(train,y)  
#print (rf1.oob_score_)#0.99808
#再对 内部节点再划分所需最小样本数min_samples_split和叶子节点最少样本数min_samples_leaf一起调参
'''param_test3={'min_samples_split':[_ for _ in range(80,150,20)],'min_samples_leaf':[_ for _ in range(10,60,10)]}
gsearch3=GridSearchCV(estimator=RandomForestClassifier(n_estimators=60,max_depth=9,max_features='sqrt',oob_score=True,
                                                      random_state=10),param_grid=param_test3,scoring='roc_auc',iid=False,cv=5,n_jobs=-1)
gsearch3.fit(train,y)
print(gsearch3.grid_scores_)
print('*'*10)
print(gsearch3.best_params_)
print('*'*10)
print(gsearch3.best_score_)

[mean: 0.93066, std: 0.02593, params: {'min_samples_leaf': 10, 'min_samples_split': 80}, 
 mean: 0.93430, std: 0.02091, params: {'min_samples_leaf': 10, 'min_samples_split': 100}, 
 mean: 0.93530, std: 0.02702, params: {'min_samples_leaf': 10, 'min_samples_split': 120}, 
 mean: 0.93467, std: 0.02722, params: {'min_samples_leaf': 10, 'min_samples_split': 140}, 
 mean: 0.92919, std: 0.02519, params: {'min_samples_leaf': 20, 'min_samples_split': 80}, 
 mean: 0.93592, std: 0.02747, params: {'min_samples_leaf': 20, 'min_samples_split': 100}, 
 mean: 0.93484, std: 0.02911, params: {'min_samples_leaf': 20, 'min_samples_split': 120}, 
 mean: 0.93036, std: 0.02831, params: {'min_samples_leaf': 20, 'min_samples_split': 140}, 
 mean: 0.92417, std: 0.02707, params: {'min_samples_leaf': 30, 'min_samples_split': 80}, 
 mean: 0.93252, std: 0.02691, params: {'min_samples_leaf': 30, 'min_samples_split': 100}, 
 mean: 0.93242, std: 0.02767, params: {'min_samples_leaf': 30, 'min_samples_split': 120}, 
 mean: 0.92836, std: 0.02894, params: {'min_samples_leaf': 30, 'min_samples_split': 140}, 
 mean: 0.92965, std: 0.02582, params: {'min_samples_leaf': 40, 'min_samples_split': 80}, 
 mean: 0.92826, std: 0.02807, params: {'min_samples_leaf': 40, 'min_samples_split': 100}, 
 mean: 0.92696, std: 0.03264, params: {'min_samples_leaf': 40, 'min_samples_split': 120}, 
 mean: 0.92515, std: 0.03103, params: {'min_samples_leaf': 40, 'min_samples_split': 140}, 
 mean: 0.92452, std: 0.02529, params: {'min_samples_leaf': 50, 'min_samples_split': 80}, 
 mean: 0.92452, std: 0.02529, params: {'min_samples_leaf': 50, 'min_samples_split': 100}, 
 mean: 0.92206, std: 0.03576, params: {'min_samples_leaf': 50, 'min_samples_split': 120}, 
 mean: 0.92221, std: 0.03479, params: {'min_samples_leaf': 50, 'min_samples_split': 140}]
**********
{'min_samples_leaf': 20, 'min_samples_split': 100}
**********
0.9359221905382125
'''
#最后，我们对max_features进行调参
'''param_test4={'max_features':[_ for _ in range(2,6,1)]}
gsearch4=GridSearchCV(estimator=RandomForestClassifier(n_estimators=60,max_depth=9,min_samples_split=100,min_samples_leaf=20,
                                        oob_score=True,random_state=10),param_grid=param_test4,scoring='roc_auc',iid=False,cv=5,n_jobs=-1)
gsearch4.fit(train,y)
print(gsearch4.grid_scores_)
print('*'*10)
print(gsearch4.best_params_)
print('*'*10)
print(gsearch4.best_score_)

[mean: 0.93592, std: 0.02747, params: {'max_features': 2}, 
 mean: 0.92505, std: 0.02320, params: {'max_features': 3}, 
 mean: 0.92823, std: 0.01968, params: {'max_features': 4}, 
 mean: 0.91520, std: 0.02424, params: {'max_features': 5}]
**********
{'max_features': 2}
**********
0.9359221905382125
'''
#查看缺失值有多少
'''print(train.isnull().sum())
print('*'*10)
print(test.isnull().sum())'''
#查看不一样的非空值有多少
'''cols=['ip','app','device','os','channel']
uniques_train={col:train[col].nunique() for col in cols}
print('Train:Unique Values')
uniques_train'''
'''uniques_test={col:test[col].nunique() for col in cols}
print('Test:Unique Values')
uniques_test'''
y=train['is_attributed']
#'click_time','is_attributed','attributed_timed'
train.drop(['click_time','is_attributed','attributed_timed'],axis=1,inplace=True)#inplace=True代表更改原内存的值
#'click_id','click_time'
test.drop(['click_id','click_time'],axis=1,inplace=True)
#不包含时间
#train.drop(['click_time'],axis=1,inplace=True)
#test.drop(['click_time'],axis=1,inplace=True)
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
#建立模型开始训练
#rfm=RandomForestClassifier(n_estimators=12,max_depth=6,min_samples_leaf=100,max_features=0.5,bootstrap=False,n_jobs=-1,random_state=123)
#%time rfm.fit(train,y)
#用我们搜索到的最佳参数，我们再看看最终的模型拟合
rf2=RandomForestClassifier(n_estimators=15,max_depth=9,min_samples_split=100,min_samples_leaf=20,max_features=2,
                           oob_score=True,random_state=10,n_jobs=-1)
rf2.fit(train,y)
print(rf2.oob_score_)#0.99808
#计算准确率，交叉验证法
import scikitplot.plotters as skplt
#print_score(rfm,train,y)#[0.99808]
print_score(rf2,train,y)#[0.99808]
#查看训练各个特征的权重
cols=train.columns
Iml=rf2.feature_importances_
feature_imp_dict = {}
for i in range(len(cols)):
    feature_imp_dict[cols[i]]=Iml[i]
print(feature_imp_dict)
#预测值
y_pred=rf2.predict_proba(test)
test_submission['is_attributed']=y_pred[:,1]
test_submission.head()

test_submission.head(30)
test_submission['is_attributed'].sum()
#保存结果
test_submission.to_csv('rf3_result.csv',index=False)