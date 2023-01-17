#coding:utf-8
from pandas import Series,DataFrame
import matplotlib.pyplot as plt

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
from sklearn.ensemble import RandomForestRegressor



#1.data load
data_train = pd.read_csv("../input/train.csv")
# data_train
# data_train.info()
# data_train.describe()



#2.data exploration in visualization
# 数据各属性自分布
fig = plt.figure(figsize=(20,10))
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

plt.subplot2grid((2,3),(0,0))             # 在一张大图里分列几个小图
data_train.Survived.value_counts().plot(kind='bar')# 柱状图 
plt.title("Survived (denoted in 1)") # 标题
plt.ylabel("Num")  

plt.subplot2grid((2,3),(0,1))
data_train.Pclass.value_counts().plot(kind="bar")
plt.ylabel("Num")
plt.title("Pclass distribution")

plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel("Age")                         # 设定纵坐标名称
plt.grid(b=True, which='major', axis='y') 
plt.title("Age in Survived(1) ")

plt.subplot2grid((2,3),(1,0),colspan=2)
data_train.Age[data_train.Pclass==1].plot(kind='kde')
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel("Age")# plots an axis lable
plt.ylabel("probability") 
plt.title("age distribution in Pclss")
plt.legend(('Pclass=1','Pclass=2','Pclass=3'),loc='best')

plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title("Embarked distribution")
plt.ylabel("Num")  
plt.show()

# 数据各属性与结果分布
fig = plt.figure()
fig.set(alpha=0.2)
Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
# print('Pclass distribution of survived_0:\n',Survived_0,'\nPclass distribution of survived_0:\n',Survived_0)
df = pd.DataFrame({'survived_1': Survived_1, 'survived_0': Survived_0})
df.plot(kind='bar', stacked=True)
plt.title('survived for all Pclass')
plt.xlabel('Pclass')
plt.ylabel('Num')
plt.grid(True)
plt.show()

fig = plt.figure()
fig.set(alpha=0.2)
Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
df = pd.DataFrame({'male':Survived_m,'female': Survived_f})
print(df)
df.plot(kind='bar', stacked=True)
plt.title('survived for  sex')
plt.xlabel('SEX')
plt.ylabel('Num')
plt.show()

fig = plt.figure(figsize=(20,5))
fig.set(alpha=0.2)
plt.title('survived in sex and Pclass')
ax1 = fig.add_subplot(1, 4,1)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar',label='female of high class',color='#fa2479')
ax1.set_xticklabels(['sur', 'unsur'], rotation=0)
ax1.legend(['female of high class'], loc='best')

ax2=fig.add_subplot(142, sharey=ax1)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')
ax2.set_xticklabels(["unsur", "sur"], rotation=0)
plt.legend(['female of low class'], loc='best')

ax3=fig.add_subplot(143, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')
ax3.set_xticklabels(["unsur", "sur"], rotation=0)
plt.legend(['male of high class'], loc='best')

ax4=fig.add_subplot(144, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male low class', color='steelblue')
ax4.set_xticklabels(["unsur", "sur"], rotation=0)
plt.legend(["male of low class"], loc='best')
plt.show()

g = data_train.groupby(['SibSp','Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
#print(df)




#3.loss data processing
# 3.1sklearn-RandomForestRegressor-'Age'
def missing_age(df):
    age_df = df[['Age','Fare','Parch','SibSp','Pclass']]
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age=age_df[age_df.Age.isnull()].as_matrix()
    print('known_age:\n',known_age)
    #RandomForestRegressor
    y = known_age[:, 0]# y即目标年龄'Age'
    X = known_age[:, 1:]  # X即特征属性值'Fare','Parch','SibSp','Pclass'
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)
    Ages_predected=rfr.predict(unknown_age[:,1:])
    df.loc[(age_df.Age.isnull()),'Age']=Ages_predected#预测值补回原df
    return df,rfr
#Yes Or No-'Cabin'
def set_cabin_type(df):
    df.loc[(df.Cabin.isnull()),'Cabin']='Yes'
    df.loc[(df.Cabin.notnull()),'Cabin']='No'
    return df

data_train,rfr=missing_age(data_train) 
data_train=set_cabin_type(data_train)
#print(data_train)
#print(data_train['Age'])

#3.pd.get_dummies-'Cabin....'
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')
df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
#print(df)

#StandardScaler
import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()
age_fare=df.as_matrix(['Age','Fare'])
scaler.fit(age_fare)  
scaled_age_fare=scaler.transform(age_fare)  
df['Age_scaled']=scaled_age_fare[:,0]  
df['Fare_scaled']=scaled_age_fare[:,1]  
df  

#4. 把需要的feature字段取出来，转成numpy格式，训练模型
from sklearn import linear_model
# 用正则取出我们要的属性值
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()
# y即Survival结果
y = train_np[:, 0]
# X即特征属性值
X = train_np[:, 1:]
# fit到RandomForestRegressor之中
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X, y)
print('\nmodel:\n',clf)

# test_data 相同处理
data_test = pd.read_csv("../input/test.csv")
data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0
# 接着我们对test_data做和train_data中一致的特征变换
# 首先用同样的RandomForestRegressor模型填上丢失的年龄
tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()
# 根据特征属性X预测年龄并补上
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges

data_test = set_cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')


df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
age_fare_test=df_test.as_matrix(['Age','Fare'])
scaler.fit(age_fare_test)  
scaled_age_fare_test=scaler.transform(age_fare_test)  
df_test['Age_scaled']=scaled_age_fare_test[:,0]  
df_test['Fare_scaled']=scaled_age_fare_test[:,1]  
df_test

#开始预测
test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv('submission.csv', index=False)
print('\nresults:\n',result)


