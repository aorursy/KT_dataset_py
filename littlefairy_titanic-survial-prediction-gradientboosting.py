# 忽略警告提示

import warnings

warnings.filterwarnings('ignore')



# 导入数据处理包

import pandas as pd

import numpy as np
# 导入训练数据集

train = pd.read_csv('../input/titanic/train.csv')

# 导入测试数据集

test = pd.read_csv('../input/titanic/test.csv')
# 查看训练数据内容

train.head()
# 查看测试数据内容

test.head()
# 查看数据大小

print('训练数据集：',train.shape,'测试数据集：',test.shape)
# 查看数据的基本信息

train.info()
test.info()
# 合并数据集，方便对两个数据集进行同步清洗

full = train.append(test,ignore_index=True)

print('合并后的数据集：',full.shape)
# 获取描述统计信息

full.describe()
# 查看数据缺失情况

full.info()
# 导入分析包

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# 提取数值型特征

df_num = pd.DataFrame(train,columns=['Age','Fare','SibSp','Parch','Survived'])

# 绘制特征分布图

df_num.hist(figsize=(25,20),bins=20,xlabelsize=15,ylabelsize=15)
# 提取类别型特征

df_category1 = pd.DataFrame(train,columns=['Sex','Embarked','Pclass','Survived'])

# 绘制类别分布图

fig = plt.figure(figsize=(25,7))

plt.subplot(131)

df_category1.groupby('Sex').agg('count')['Survived'].plot(kind='bar',fontsize=15)

plt.subplot(132)

df_category1.groupby('Embarked').agg('count')['Survived'].plot(kind='bar',fontsize=15)

plt.subplot(133)

df_category1.groupby('Pclass').agg('count')['Survived'].plot(kind='bar',fontsize=15)
# Name提取头衔作为类别

def getTitle(name):

    '''

    从name中提取头衔

    '''

    str1 = name.split(',')[1]

    str2 = str1.split('.')[0].strip()

    

    return str2

df_title = train['Name'].map(getTitle)

df_title.head()
# Cabin提取首字母作为类别(缺失数据为n)

df_Cabin = train['Cabin'].astype(str).map(lambda x:x[0])

df_Cabin.head()
# 查看Ticket取值

train['Ticket'].value_counts()
# Tickets提取类别

def getTicket(ticket):

    ticket = ticket.replace('.',' ')

    ticket = ticket.replace('/',' ')

    ticket = ticket.split(' ')

    ticket = map(lambda a:a.strip(),ticket)

    ticket = list(filter(lambda x:not x.isdigit(),ticket))

    if len(ticket)>0:

        return ticket[0]

    else:

        return 'Unknown'

df_tickets = train['Ticket'].map(getTicket)

df_tickets.value_counts()
# # 提取类别型特征

df_category2 = pd.concat([df_title,df_Cabin,df_tickets,full['Survived']],axis=1)

# 绘制类别分布图

fig = plt.figure(figsize=(25,7))

plt.subplot(131)

df_category2.groupby('Name').agg('count')['Survived'].plot(kind='bar',fontsize=15)

plt.subplot(132)

df_category2.groupby('Cabin').agg('count')['Survived'].plot(kind='bar',fontsize=15)

plt.subplot(133)

df_category2.groupby('Ticket').agg('count')['Survived'].plot(kind='bar',fontsize=15)
# 查看不同性别的生存率

fig = plt.figure(figsize=(15,7))

df_category1.groupby('Sex').agg('mean')['Survived'].plot(kind='bar',fontsize=15,stacked=True)

plt.xlabel('Sex',fontsize=15)

plt.ylabel('survival rate',fontsize=15)



plt.show()
# 查看不同船舱等级的生存率

fig=plt.figure(figsize=(18,7))

df_category1.groupby('Pclass').agg('mean')['Survived'].plot(kind='barh',fontsize=15)

plt.ylabel('Pclass',fontsize=15)

plt.xlabel('survival rate',fontsize=15)

plt.show()
# 进行年龄段划分,查看每个年龄段的生存率

def get_agePeriod(age):

    if not np.bool(age):

        return age

    x = np.linspace(0,80,9)

    for i in range(0,len(x)-1):

        if (age>x[i] and age<=x[i+1]):

            return '(%.0f,%.0f]' %(x[i],x[i+1])

    

    

df_num['Age'] = df_num['Age'].map(get_agePeriod)    

df_num['Age'].head()
fig = plt.figure(figsize=(15,7))

df_num.groupby('Age').agg('mean')['Survived'].plot(kind='bar',fontsize=15)

plt.ylabel('Age Period',fontsize=15)

plt.xlabel('survival rate',fontsize=15)

plt.show()
train['Survived'].value_counts()
# 年龄（Age）缺失值填充

full['Age'] = full['Age'].fillna(full['Age'].median())

# 船票价格（Fare）缺失值填充

full['Fare'] = full['Fare'].fillna(full['Age'].median())
# 登船港口（Embarked）缺失值填充

'''

将缺失值填充为最频繁出现的值

'''

pd.value_counts(full['Embarked'])
full['Embarked'] = full['Embarked'].fillna('S')
# 船舱号（Cabin）缺失值填充

'''

缺失值太多，船舱号缺失值填充为‘U’，表示未知

'''

full['Cabin'] = full['Cabin'].fillna('U')
# 查看缺失处理后的数据

full.info()
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler
# Age的分布接近正态分布,因此进行0-1归一化处理

standard_scaler = StandardScaler()

full['Age']= pd.DataFrame(standard_scaler.fit_transform(full['Age'].values.reshape(-1,1)))
# Fare的分布不是正态分布，进行minmax标准化处理

min_max_scaler = MinMaxScaler()

full['Fare']= pd.DataFrame(min_max_scaler.fit_transform(full['Fare'].values.reshape(-1,1)))

full.head()
'''

将同代直系亲属数SibSp和不同代直系亲属数Parch组成组合特征家庭成员数Family

FamilySize = SibSp+Parch+1(自己)

'''

# 存放家庭成员数

familyDf = pd.DataFrame()

familyDf['FamilySize'] = full['SibSp'] + full['Parch'] + 1

familyDf['FamilySize'].describe()
'''

将家庭成员数映射至家庭类别：

小家庭：家庭成员数=1

中等家庭：2<=家庭成员数<=4

大家庭：家庭成员数>=5

'''

# 映射至家庭类别

familyDf['Family_Single'] = familyDf['FamilySize'].map(lambda s:1 if s==1 else 0)

familyDf['Family_Small'] = familyDf['FamilySize'].map(lambda s:1 if s>=2 and s<=4 else 0)

familyDf['Family_Large'] = familyDf['FamilySize'].map(lambda s:1 if s>=5 else 0)

familyDf.head()
# 将组合特征和分级特征添加至泰坦尼克号数据集

full = pd.concat([full,familyDf],axis=1)

# 删除原始SibSp和Parch列

full.drop(['SibSp','Parch'],axis=1,inplace=True)

full.head()
# Name

# one-hot编码

'''

将性别映射为数值

男（male）对应数值1，女（female）对应数值0

'''

sex_mapDict = {'male':1,'female':0}

full['Sex'] = full['Sex'].map(sex_mapDict)

full['Sex'].head()
# Embarked

# 1.存放提取后的特征

embarkedDf = pd.DataFrame()

# 2.使用get_dummies进行one-hot编码，列明前缀为Embarked

embarkedDf = pd.get_dummies(full['Embarked'],prefix='Embarked')

embarkedDf.head()
# 添加one-hot编码后的虚拟变量（dummy variables）到泰坦尼克号数据集full

full = pd.concat([full,embarkedDf],axis=1)

# 删除one-hot编码前的Embarked列

full.drop(['Embarked'],axis=1,inplace=True)

full.head()
# Pclass

# 1.存放提取后的特征

pclassDf = pd.DataFrame()

# 2.使用get_dummies进行one-hot编码，列明前缀为Pclass

pclassDf = pd.get_dummies(full['Pclass'],prefix='Pclass')

pclassDf.head()
# 添加one-hot编码后的虚拟变量（dummy variables）到泰坦尼克号数据集full

full = pd.concat([full,pclassDf],axis=1)

# 删除one-hot编码前的Pclass列

full.drop(['Pclass'],axis=1,inplace=True)

full.head()
# Name

# 提取title

# 1.存放提取后的特征

titleDf = pd.DataFrame()

# 2.使用map函数：对Seris的每个数据应用自定义的函数

titleDf['Title'] = full['Name'].map(getTitle)

titleDf.head()
# 查看称谓类别

titleDf['Title'].value_counts()
# 设置title和头衔的映射字典

title_mapDict = {'Capt':'Officer',

                'Col':'Officer',

                'Major':'Officer',

                'Jonkheer':'Royalty',

                'Don':'Royalty',

                'Sir':'Royalty',

                'the Countess':'Royalty',

                 'Dona':'Royalty',

                 'Dr':'Officer',

                 'Rev':'Officer',

                 'Lady':'Royalty',

                 'Mr':'Mr',

                 'Miss':'Miss',

                 'Mrs':'Mrs',

                 'Master':'Master',

                 'Mlle':'Miss',

                 'Mme':'Mrs',

                 'Ms':'Mrs'

                }
# 使用map函数进行映射

titleDf['Title'] = titleDf['Title'].map(title_mapDict)

# 使用get_dummies进行one-hot编码

titleDf = pd.get_dummies(titleDf['Title'])

titleDf.head()
# 添加one-hot编码后的虚拟变量（dummy variables）到泰坦尼克号数据集full

full = pd.concat([full,titleDf],axis=1)

# 删除one-hot编码前的name列

full.drop(['Name'],axis=1,inplace=True)

full.head()
# 查看船舱号内容

full['Cabin'].head()
# Cabin

cabinDf = pd.DataFrame()

'''

客舱号的类别值是首字母，例如：

C85 类别映射为首字母C

'''

full['Cabin'] = full['Cabin'].map(lambda c:c[0])



# 使用get_dummuies进行one-hot编码，列名前缀为Cabin

cabinDf = pd.get_dummies(full['Cabin'],prefix='Cabin')

cabinDf.head()
# 添加one-hot编码后的虚拟变量（dummy variables）到泰坦尼克号数据集

full = pd.concat([full,cabinDf],axis=1)

# 删除one-hot编码前的Cabin列

full.drop(['Cabin'],axis=1,inplace=True)

full.head()
# Ticket

full['Ticket'] = full['Ticket'].map(getTicket)

# one_hot编码

ticketDf = pd.get_dummies(full['Ticket'],prefix='Ticket')

full = pd.concat([full,ticketDf],axis=1)

full.drop(['Ticket'],axis=1,inplace=True)

full.head()
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectFromModel
# 查看特征数量

full.shape
# 删除PassengerId列

full_noid = full.drop(['PassengerId'],axis=1)
# 还原训练集和测试集

def reverse(full):

    full_c = full.copy(deep=True)

    type(full_c)

    # 原始数据集共有891行

    train_num= 891

    # 存储原始数据集：特征

    train_X = full_c.drop(columns ='Survived',axis=1).loc[0:train_num-1,:]

    # 存储原始数据集：标签

    train_y = full.loc[0:train_num-1,'Survived']

    # 预测数据集

    pred_X = full_c.drop(columns ='Survived',axis=1).loc[train_num:,:]

    

    return train_X,train_y,pred_X
train_X,train_y,pred_X = reverse(full_noid)
print(train_X.shape,train_y.shape,pred_X.shape)
clf = RandomForestClassifier(n_estimators=100,max_features='sqrt')

clf = clf.fit(train_X,train_y)
# 查看feature importance 

features =pd.DataFrame()

features['feature']=train_X.columns

features['importance'] = clf.feature_importances_

features.sort_values(by='importance',ascending=True,inplace=True)

features.set_index(['feature'],inplace=True)

features.plot(kind='barh',figsize=(20,20))
slf = SelectFromModel(clf,prefit=True)

train_X_new = slf.transform(train_X)

print('特征选择前训练特征个数：',train_X.shape[1],

     '\n特征选择后训练特征个数：',train_X_new.shape[1])

pred_X_new = slf.transform(pred_X)

print('特征选择前预测特征个数：',pred_X.shape[1],

     '\n特征选择后预测特征个数：',pred_X_new.shape[1])
'''

确保原始数据集大小为891，且与预测数据集维数相同，以防止构建模型时报错

'''

print('原始数据集大小：',train_X_new.shape)

print('预测数据集大小：',pred_X_new.shape)
'''

拆分训练数据和测试数据

'''

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(train_X_new,train_y,train_size=.8)

print('训练数据集特征:',X_train.shape,

     '测试数据集特征：',X_test.shape,

     )

print('训练数据集标签：',y_train.shape,

     '测试数据集标签：',y_test.shape)
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier 

from sklearn.svm import  SVC

from xgboost import XGBClassifier,XGBRFClassifier

from sklearn.model_selection import cross_val_score
lr = LogisticRegression()

rfc = RandomForestClassifier()

gboosting = GradientBoostingClassifier()

svc = SVC()

xgb = XGBClassifier()

xgbrfc = XGBRFClassifier()



models = [lr,rfc,gboosting,svc,xgb,xgbrfc]
# 模型选择

for model in models:

    print('Cross validation of:{0}'.format(model.__class__))

    score = cross_val_score(model,X_train,y_train,cv=5,scoring='accuracy')

    score = score.mean()

    print('score = {0}'.format(score))
from sklearn.model_selection import GridSearchCV
param_test1 = {

    'n_estimators':range(20,81,10)

}

# 训练参数

gsearch1 = GridSearchCV(estimator=GradientBoostingClassifier(

    learning_rate=0.1,

    max_depth=6,

    min_samples_split=5,

    min_samples_leaf=2,

    max_features='sqrt',

    subsample=0.8,

    random_state=10

    ),param_grid=param_test1,scoring="accuracy",n_jobs=4,iid=False,cv=5

)

gsearch1.fit(X_train,y_train)

n_estimators = gsearch1.best_params_['n_estimators']

print(gsearch1.scoring) 

print(gsearch1.best_params_)

print(gsearch1.best_score_) 
param_test2 = {

    'max_depth':range(2,8),

    'min_samples_split':range(2,10)

}

# 训练参数

gsearch2 = GridSearchCV(estimator=GradientBoostingClassifier(

    learning_rate=0.1,

    n_estimators=n_estimators,

    min_samples_leaf=2,

    max_features='sqrt',

    subsample=0.8,

    random_state=10

    ),param_grid=param_test2,scoring="accuracy",n_jobs=4,iid=False,cv=5

)

gsearch2.fit(X_train,y_train)

max_depth = gsearch2.best_params_['max_depth']

print(gsearch2.scoring) 

print(gsearch2.best_params_)

print(gsearch2.best_score_) 
param_test3 = {

    'min_samples_split':range(2,10),

    'min_samples_leaf':range(2,20)

}

# 训练参数

gsearch3 = GridSearchCV(estimator=GradientBoostingClassifier(

    learning_rate=0.1,

    max_depth=max_depth,

    n_estimators=n_estimators,

    max_features='sqrt',

    subsample=0.8,

    random_state=10

    ),param_grid=param_test3,scoring="accuracy",n_jobs=4,iid=False,cv=5

)

gsearch3.fit(X_train,y_train)

min_samples_split=gsearch3.best_params_['min_samples_split']

min_samples_leaf=gsearch3.best_params_['min_samples_leaf']

print(gsearch3.scoring) 

print(gsearch3.best_params_)

print(gsearch3.best_score_) 
param_test4 = {

    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]

}

# 训练参数

gsearch4 = GridSearchCV(estimator=GradientBoostingClassifier(

    learning_rate=0.1,

    max_depth=max_depth,

    n_estimators=n_estimators,

    max_features='sqrt',

    min_samples_split=min_samples_split,

    min_samples_leaf=min_samples_leaf,

    random_state=10

    ),param_grid=param_test4,scoring="accuracy",n_jobs=4,iid=False,cv=5

)

gsearch4.fit(X_train,y_train)

subsample=gsearch4.best_params_['subsample']

print(gsearch4.scoring) 

print(gsearch4.best_params_)

print(gsearch4.best_score_) 
# 设置learning_rate=0.01，此时有400个树

gbm=GradientBoostingClassifier(

    learning_rate=0.01,

    max_depth=max_depth,

    n_estimators=n_estimators*10,

    max_features='sqrt',

    min_samples_split=min_samples_split,

    min_samples_leaf=min_samples_leaf,

    subsample=subsample,

    random_state=10

)

# gbm.fit(X_train,y_train)

# print(gbm.score(X_test,y_test)) 



# 使用全部训练样本构建模型

gbm.fit(train_X_new,train_y)
# 使用机器学习模型，对预测数据集中的生存情况进行预测

pred_y = gbm.predict(pred_X_new)
pred_y
'''

预测结果为float类型，kaggle要求提交的结果为int型，

需要转换数据类型

'''

pred_y = pred_y.astype(int)
# 乘客id

passengerId = full.loc[891:,'PassengerId']

# 数据框：乘客id,预测生存情况

predDf = pd.DataFrame(

{'PassengerId':passengerId,

'Survived':pred_y})

predDf.shape
predDf.head()
# 保存结果

predDf.to_csv('titanic_pred.csv',index=False)