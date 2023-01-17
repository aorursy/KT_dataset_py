

import numpy as np # 科学计算
import pandas as pd # 数据分析
from pandas import Series,DataFrame

data_train = pd.read_csv("../input/titanic/train.csv")
data_train


data_train.info()
data_train.describe()
from matplotlib import pyplot as plt



fig = plt.figure()
fig.set(alpha=0.2) #设置图片颜色alpha参数

plt.subplot2grid((3,10),(0,0),colspan=2)   #在一张大图里分列几张小图
data_train.Survived.value_counts().plot(kind='bar') #柱状图
plt.title("survivid info(1 is suvivied)")
plt.ylabel("p number")

plt.subplot2grid((3,10),(0,4),colspan=2)
data_train.Pclass.value_counts().plot(kind="bar")
plt.ylabel("p number")
plt.title("p grade")

plt.subplot2grid((3,10),(0,8),colspan=2)
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel("age")
plt.grid(b=True,which='major', axis='y')
plt.title(" look the survived case by age")

plt.subplot2grid((10,20),(5,0),colspan=13,rowspan=5)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel('age')
plt.ylabel('density')
plt.title('distributin of p age group by grade')
plt.legend(('1st','2nd','3rd'),loc='best')

plt.subplot2grid((3,10),(2,8),colspan=2)
data_train.Embarked.value_counts().plot(kind='bar')
plt.title('p number of bark')
plt.ylabel('number')

#看看各乘客等级的获救情况
fig = plt.figure()
fig.set(alpha=0.2)#设定图表颜色alpha参数

Survived_0 = data_train.Pclass[data_train.Survived == 0 ].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df = pd.DataFrame({'survived':Survived_1, 'unsurvived':Survived_0})
df.plot(kind='bar',stacked=True)
plt.title('survived case of vary grade person')
plt.xlabel('grade of person')
plt.ylabel('number')
plt.show()
#看看各性别的获救情况
fig = plt.figure()
fig.set(alpha=0.2) #设定图表颜色alpha参数

Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
df=pd.DataFrame({'male':Survived_m, 'fmale':Survived_f})
df.plot(kind='bar',stacked=True)
plt.title('s')
plt.xlabel('sex')
plt.ylabel('number')
plt.show()
#然后我们再来看看各种舱级别情况下各性别的获救情况
fig = plt.figure(figsize=(20,8))
fig.set(alpha=0.65) #设置图像透明度，无所谓
plt.title("survived case base on Pclass^sex")

ax1 = fig.add_subplot(141)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar',label='female highclass',color='#FA2479')
ax1.legend(['female/high'],loc='best')

ax2 = fig.add_subplot(142, sharey=ax1)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar',label='female,lowclass',color='pink')
ax2.legend(['female/low'],loc='best')

ax3 = fig.add_subplot(143, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar',label='male,highclass',color='lightblue')

ax3.legend(['male/high'],loc='best')

ax4 = fig.add_subplot(144, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar',label='male,lowclass',color='steelblue')
ax4.legend(['male/low'],loc='best')
plt.show()
fig = plt.figure(figsize=(20,8))
fig.set(alpha = 0.2) #设置图表颜色alpha的参数

Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
df = pd.DataFrame({'survived':Survived_1,'unsurvived':Survived_0})
df.plot(kind='bar',stacked=True)
plt.title('the survived info all port')
plt.xlabel('enter port')
plt.ylabel('number')



g = data_train.groupby(['SibSp','Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
print(df)

g = data_train.groupby(['Parch','Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
print(df)

fig = plt.figure(figsize=(20,8))
fig.set(alpha = 0.2) #设置图表颜色alpha的参数

Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
df = pd.DataFrame({'yes':Survived_cabin,'no':Survived_nocabin}).transpose()
df.plot(kind='bar',stacked=True)
plt.title('the survived info of whether have cabin')
plt.xlabel('cabin have or not')
plt.ylabel('people number')
plt.show
from sklearn.ensemble import RandomForestRegressor

###使用RandomForestClassifier  填补缺失的年龄属性
def set_missing_ages(df):
    
    #把已有的数值类型特征取出来丢进Random Forest Regression 中
    age_df = df[['Age','Fare','Parch','SibSp','Pclass']]
    
    #乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values
    
    #y即目标年龄
    Y = known_age[:, 0]
    
    #x即特征属性值
    X = known_age[:,1:]
    
    #fit到RandomForestRegression之中
    rfr = RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)
    rfr.fit(X,Y)
    
    #用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])
    
    #用得到的预测结果填补原缺失数据
    df.loc[(df.Age.isnull()),'Age'] = predictedAges
    
    return df,rfr

def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin']='Yes'
    df.loc[(df.Cabin.isnull()),'Cabin'] = 'No'
    return df

data_train,rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)
data_train
    
dummies_Cabin = pd.get_dummies(data_train['Cabin'],prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'],prefix='Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'],prefix='Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'],prefix='Pclass')
df = pd.concat([data_train,dummies_Cabin,dummies_Embarked,dummies_Sex,dummies_Pclass],axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df
import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df['Age'].values.reshape(-1,1))
df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1,1), age_scale_param)
fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1,1))
df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1,1), fare_scale_param)
df
from sklearn import linear_model

#用正则取出我们要的属性值
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.values

#Y即是Survival结果
Y = train_np[:,0]

#X即特征属性值
X = train_np[:,1:]

#fit到RandomForestRegression之中
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6,solver='liblinear')
clf.fit(X,Y)
clf
data_test = pd.read_csv("../input/titanic/test.csv")
data_test.loc[(data_test.Fare.isnull()),'Fare'] = 0
#接着我们对test_data做和train_data中一致的特征变换
#首先用同样的randomForestRegressor模型填上丢失的年龄
tmp_df = data_test[['Age','Fare','Parch','SibSp','Pclass']]
null_age = tmp_df[data_test.Age.isnull()].values
#根据特征属性X预测年龄并补上
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges

data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')

df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].values.reshape(-1,1), age_scale_param)
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].values.reshape(-1,1), fare_scale_param)
df_test

test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
result.to_csv("predict.csv", index=True)
result

