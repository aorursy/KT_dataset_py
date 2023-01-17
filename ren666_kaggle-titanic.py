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
data_train = pd.read_csv("../input/train.csv")
data_test = pd.read_csv("../input/test.csv")
combine = [data_train,data_test]
data_train
data_train.info()
data_train.describe()
#coding:utf-8
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(30,10))
fig.set(alpha=0.2)#图表alpha参数
plt.subplot2grid((2,3),(0,0))
data_train.Survived.value_counts().plot(kind="bar")
plt.title("The number of surviving(1 for survived)")
plt.ylabel("numbers of passengers")

plt.subplot2grid((2,3),(0,1))
data_train.Pclass.value_counts().plot(kind="bar")
plt.title("the Pclass of passengers")
plt.ylabel("numbers of passengers")

plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel(u"age")                         # 设定纵坐标名称
plt.grid(b=True, which='major', axis='y') 
plt.title(u"survived by age (1 for survived)")

plt.subplot2grid((2,3),(1,0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')   
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel(u"age")# plots an axis lable
plt.ylabel(u"density") 
plt.title(u"passengers' age by Pclass")
plt.legend((u'top', u'2',u'3'),loc='best') # sets our legend for our graph.

plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title(u"the number embarked from differet harbors")
plt.ylabel(u"numbers")

plt.show()
#不同性别对是否获救的影响
fig = plt.figure()
fig.set(alpha=0.2)

Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
df = pd.DataFrame({'female':Survived_f,'male':Survived_m})
df.plot(kind="bar",stacked="True")
plt.title("Survived by different sex")
plt.xlabel("sex")
plt.ylabel("numbers")

plt.show()
#不同舱等级对是否获救的影响
fig = plt.figure()
fig.set(alpha=0.2)
Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df = pd.DataFrame({'Survived':Survived_1,'not Survived':Survived_0})
df.plot(kind="bar",stacked="True")
plt.title("Survived by different class")
plt.xlabel("class")
plt.ylabel("numbers")

plt.show()


#不同舱级别男女获救情况
fig = plt.figure(figsize=(30,20))
fig.set(alpha=0.65)
plt.title("Survived by class and by age")

ax1=fig.add_subplot(141)
S1=data_train.Survived[data_train.Sex=='female'][data_train.Pclass != 3].value_counts()
S1.plot(kind='bar',label='female highclass',color='#FA2479')
ax1.set_xticklabels([u"Survived", u"not Survived"], rotation=0)
ax1.legend([u"female/high class"], loc='best')

ax2=fig.add_subplot(142)
S2=data_train.Survived[data_train.Sex=='male'][data_train.Pclass != 3].value_counts()
S2.plot(kind='bar',label='male highclass',color='pink')
ax2.set_xticklabels([u"Survived", u"not Survived"], rotation=0)
ax2.legend([u"male/high class"], loc='best')

ax3=fig.add_subplot(143)
S3=data_train.Survived[data_train.Sex=='female'][data_train.Pclass == 3].value_counts()
S3.plot(kind='bar',label='female lowclass',color='lightblue')
ax3.set_xticklabels([u"Survived", u"not Survived"], rotation=0)
ax3.legend([u"female/low class"], loc='best')

ax4=fig.add_subplot(144)
S4=data_train.Survived[data_train.Sex=='male'][data_train.Pclass == 3].value_counts()
S4.plot(kind='bar',label='male lowclass',color='steelblue')
ax4.set_xticklabels([u"Survived", u"not Survived"], rotation=0)
ax4.legend([u"male/low class"], loc='best')


#不同港口对是否获救的影响
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'Survived':Survived_1, u'not Survived':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"Survived by different harbors")
plt.xlabel(u"harbor") 
plt.ylabel(u"numbers") 

plt.show()
#
g = data_train.groupby(['SibSp','Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
print(df)

g = data_train.groupby(['Parch','Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
print(df)
data_train.Cabin.value_counts()
#使用 RandomForestClassifier 填补缺失的年龄属性
from sklearn.ensemble import RandomForestRegressor
def get_rfr(df):

    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].values
#     unknown_age = age_df[age_df.Age.isnull()].values
    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)
    return rfr

    
def set_missing_ages(rfr,df):
    df.loc[(df.Fare.isnull(),'Fare')]=0
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
    unknown_age = age_df[age_df.Age.isnull()].values    
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 
    return df

    

# def set_Cabin_type(df):
#     df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
#     df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
#     return df
# data_train = set_Cabin_type(data_train)
rfr = get_rfr(data_train)
data_train = set_missing_ages(rfr,data_train)
data_test = set_missing_ages(rfr,data_test)

#港口登陆信息train_data缺失2个,用S(最通用)填补
data_train.loc[data_train.Embarked.isnull(),'Embarked']='S'
data_train.info()
combine = [data_train,data_test]
data_train.info()
data_test.info()
#加一个child的特征 年龄小于14岁
combine=[data_train,data_test]

# for dataset in combine:
#     dataset['isChild']=0
#     dataset.loc[dataset['Age']<=14,'isChild']=1
# data_train.shape
#融合 SibSp和PArch为一个famliysize特征
for dataset in combine:
    dataset['FamliySize']=dataset['SibSp']+dataset['Parch']+1
data_train.head()
for dataset in combine:
    dataset.loc[dataset['Sex']=='male','Sex'] = 0 
    dataset.loc[dataset['Sex']=='female','Sex'] = 1
data_train.head()
#加一个isAlone特征
for dataset in combine:
    dataset['isAlone']=0
    dataset.loc[dataset['FamliySize']==1,'isAlone'] = 1    
data_train
#年龄分段划分0-80岁
for dataset in combine:
    dataset['AgeBand']='0'
    dataset.loc[(dataset['Age']>0)&(dataset['Age']<16),'AgeBand']=0
    dataset.loc[(dataset['Age']>=16)&(dataset['Age']<32),'AgeBand']=1
    dataset.loc[(dataset['Age']>=32)&(dataset['Age']<48),'AgeBand']=2
    dataset.loc[(dataset['Age']>=48)&(dataset['Age']<64),'AgeBand']=3
    dataset.loc[(dataset['Age']>=64),'AgeBand']=4
data_train.info()
    
# #特征因子化 data_train
# dummies_Embarked  = pd.get_dummies(data_train['Embarked'],prefix='Embarked')

# dummies_Sex = pd.get_dummies(data_train['Sex'],prefix='Sex')

# dummies_Pclass = pd.get_dummies(data_train['Pclass'],prefix = 'Pclass')

# dummies_isChild = pd.get_dummies(data_train['isChild'],prefix = 'isChild')

# dummies_FamliySize =pd.get_dummies(data_train['FamliySize'],prefix = 'FamliySize')

# dummies_isAlone =pd.get_dummies(data_train['isAlone'],prefix = 'isAlone')

# dummies_AgeBand =pd.get_dummies(data_train['AgeBand'],prefix = 'AgeBand')

# data_train = pd.concat([data_train,dummies_Embarked,dummies_Sex,dummies_Pclass,dummies_isChild,dummies_FamliySize,dummies_isAlone,dummies_AgeBand],axis=1)
    
# data_train.head()
# #特征因子化 data_test
# dummies_Embarked  = pd.get_dummies(data_test['Embarked'],prefix='Embarked')

# dummies_Sex = pd.get_dummies(data_test['Sex'],prefix='Sex')

# dummies_Pclass = pd.get_dummies(data_test['Pclass'],prefix = 'Pclass')

# dummies_isChild = pd.get_dummies(data_test['isChild'],prefix = 'isChild')

# dummies_FamliySize =pd.get_dummies(data_test['FamliySize'],prefix = 'FamliySize')

# dummies_isAlone =pd.get_dummies(data_test['isAlone'],prefix = 'isAlone')

# dummies_AgeBand =pd.get_dummies(data_test['AgeBand'],prefix = 'AgeBand')

# data_test = pd.concat([data_test,dummies_Embarked,dummies_Sex,dummies_Pclass,dummies_isChild,dummies_FamliySize,dummies_isAlone,dummies_AgeBand],axis=1)
    
# data_test.info()
data_train.drop(['PassengerId','Name','Sex','Age','SibSp','Parch','Ticket','Cabin','Embarked'],axis=1,inplace=True)
data_test.drop(['Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Cabin','Embarked',], axis=1, inplace=True)

train_df = data_train
test_df = data_test
# #归一化
# import sklearn.preprocessing as preprocessing
# scaler = preprocessing.StandardScaler()
# age_scale_param = scaler.fit(train_df['Age'].values.reshape(-1,1))
# train_df['Age_scaled']= scaler.fit_transform(train_df['Age'].values.reshape(-1,1),age_scale_param)
# fare_scale_param = scaler.fit(train_df['Fare'].values.reshape(-1,1))
# train_df['Fare_scaled'] = scaler.fit_transform(train_df['Fare'].values.reshape(-1,1),fare_scale_param)
# train_df
# test_df['Age_scaled'] = scaler.fit_transform(test_df['Age'].values.reshape(-1,1), age_scale_param)
# test_df['Fare_scaled'] = scaler.fit_transform(test_df['Fare'].values.reshape(-1,1), fare_scale_param)
# test_df
#使用sklearn里的bagging 模型融合
from sklearn.ensemble import BaggingRegressor
from sklearn import linear_model

# train_df = train_df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*')
# train_np = train_df.values
y = train_df.values[:,0]
X = train_df.values[:,1::]
clf = linear_model.LogisticRegression(C=1.0,penalty='l1',tol=1e-6)
bagging_clf = BaggingRegressor(clf,n_estimators=20,max_samples=0.8,n_jobs=-1)
bagging_clf.fit(X,y)

# test = test_df.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = bagging_clf.predict(test_df)
result = pd.DataFrame({'PassengerId':test_df['PassengerId'].values,'Survived':predictions.astype(np.int32)})
result.to_csv('bagging_submission.csv',index = False)
#使用sklearn中得到LogisticsRegression建模
from sklearn import linear_model
y = train_df.values[:,0]
X = train_df.values[:,1::]

clf= linear_model.LogisticRegression(C=1.0,penalty='l1',tol=1e-6)
clf.fit(X,y)
clf

# test = test_df.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*')
# predictions = clf.predict(test)
# result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values,'Survived':predictions.astype(np.int32)})
# result
# result.to_csv('Submission.csv',index=False)
pd.DataFrame({"columns":list(train_df.columns)[1:],"coef":list(clf.coef_.T)})
# #查看打分情况
# from sklearn import cross_validation
# clf =linear_model.LogisticRegression(C=1.0,penalty='l1',tol=1e-6)
# all_data = train_df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*')
# X = all_data.values[:,1:]
# y = all_data.values[:,0]
# print (cross_validation.cross_val_score(clf,X,y,cv=5))

# #分割数据 训练数据:cv数据=7:3
# split_train,split_cv = cross_validation.train_test_split(train_df,test_size=0.3,random_state=0)
# split_train_df =split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Embarked_,*|Sex_.*|Pclass_.*')
# #生成,模型
# clf = linear_model.LogisticRegression(C=1.0,penalty='l1',tol=1e-6)
# X =split_train_df.values[:,1:]
# y = split_train_df.values[:,0]
# clf.fit(X,y)
# #对cross validation数据进行预测
# split_cv_df = split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*')
# predictions = clf.predict(split_cv_df.values[:,1:])

# origin_data_train = pd.read_csv("../input/train.csv")
# bad_cases = origin_data_train.loc[origin_data_train['PassengerId'].isin(split_cv[predictions!=split_cv_df.values[:,0]]['PassengerId'].values)]
# bad_cases

# # 根据学习曲线判断模型状态 过拟合 欠拟合
# import numpy as np 
# import matplotlib.pyplot as plt
# from sklearn.learning_curve import learning_curve

# #用sklearn的learning_curve得到training_score, 使用matplot画出learning curve
# def plot_learning_curve(estimator,title,X,y,ylim=None,cv=None,n_jobs=1,train_sizes=np.linspace(.05,1.,20),verbose=0,plot=True):
#     """
#     画出data在某模型上的learning curve
#     参数说明:
#     ----------------------------------
#     estimateor:你用的分类器
#     title: 标题
#     X : 输入的feature,numpy类型
#     y : 输入的target 
#     ylim  : tuple格式的(ymin,ymax),设定图像中坐标的最高点和最低点
#     cv :做cross validation的时候,数据分成的份数 ,默认是3份
#     n_jobs:并行的任务数(默认1)
#     """
#     train_sizes,train_scores,test_scores = learning_curve(estimator,X,y,cv=cv,n_jobs=n_jobs,train_sizes=train_sizes,verbose=verbose)
#     train_scores_mean  = np.mean(train_scores,axis=1)
#     train_scores_std = np.std(train_scores,axis=1)
#     test_scores_mean = np.mean(test_scores,axis=1)
#     test_scores_std = np.std(test_scores,axis=1)
    
#     if plot:
#         plt.figure()
#         plt.title(title)
#         if ylim is not None:
#             plt.ylim(*ylim)
#         plt.xlabel('train samples')
#         plt.ylabel('score')
# #         plt.gca().invert_yaxis()
#         plt.grid()
        
#         plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 
#                          alpha=0.1, color="b")
#         plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 
#                          alpha=0.1, color="r")
#         plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"scores on train dataset")
#         plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"scores on cv dataset")

#         plt.legend(loc="best")

#         plt.draw()
#         plt.show()
# #         plt.gca().invert_yaxis()

#     midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
#     diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
#     return midpoint, diff
        
# plot_learning_curve(clf, u"learning curve", X, y)
