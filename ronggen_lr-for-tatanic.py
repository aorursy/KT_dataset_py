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
#导入库
import pandas as pd 
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt 
%matplotlib inline
data_train = pd.read_csv('../input/train.csv')
data_train.head()
#观察各个特征与获救情况的关系
import matplotlib.pyplot as plt
#看性别与获救情况的关系
fig = plt.figure(figsize=(10,10))

fig.set(alpha=0.2)  # 设定图表颜色alpha参数

ax1 = plt.subplot(2,2,1)
#plt.subplot2grid((2,3),(0,0))             # 在一张大图里分列几个小图

UnSurvived = data_train.loc[data_train['Survived']==0,'Sex'].value_counts()
Survived = data_train.loc[data_train['Survived']==1,'Sex'].value_counts()
df = pd.DataFrame({'UnSurvived':UnSurvived,'Survived':Survived})
df.plot(kind='bar',stacked=True,ax=ax1)
plt.xlabel('Sex')
plt.ylabel('number')

#看等级与获救情况的关系
ax2 = plt.subplot(2,2,2)
UnSurvived = data_train.loc[data_train['Survived']==0,'Pclass'].value_counts()
Survived = data_train.loc[data_train['Survived']==1,'Pclass'].value_counts()
df = pd.DataFrame({'UnSurvived':UnSurvived,'Survived':Survived})
df.plot(kind='bar',stacked=True,ax=ax2)
plt.xlabel('Pclass')

#看登船口岸与获救情况的关系
ax3 = plt.subplot(2,2,3)
UnSurvived = data_train.loc[data_train['Survived']==0,'Embarked'].value_counts()
Survived = data_train.loc[data_train['Survived']==1,'Embarked'].value_counts()
df = pd.DataFrame({'UnSurvived':UnSurvived,'Survived':Survived})
df.plot(kind='bar',stacked=True,ax=ax3)
plt.xlabel('Embarked')

#查看父母与小孩个数 与获救情况的关系
ax4 = plt.subplot(2,2,4)
UnSurvived = data_train.loc[data_train['Survived']==0,'Parch'].value_counts()
Survived = data_train.loc[data_train['Survived']==1,'Parch'].value_counts()
df = pd.DataFrame({'UnSurvived':UnSurvived,'Survived':Survived})
df.plot(kind='bar',stacked=True,ax=ax4)
plt.xlabel('Parch')

plt.show()
#解决一下缺失值问题（参考寒）

#看一下Cabin，有无对获救情况的影响
data_train.loc[data_train['Cabin'].notnull(),'Cabin'] = 'Yes'
data_train.loc[data_train['Cabin'].isnull(),'Cabin'] = 'No'
#观察一下分布
Survived = data_train.loc[data_train['Survived']==1,'Cabin'].value_counts()
UnSurvived = data_train.loc[data_train['Survived']==0,'Cabin'].value_counts()
df = pd.DataFrame({'UnSurvived':UnSurvived,'Survived':Survived})
df.plot(kind='bar',stacked=True)
#对年龄进行缺失值拟合
#先把年龄取平均值看看
import numpy as np

data_train.loc[:,'Age'].fillna(np.mean(data_train['Age']),inplace=True)

Age_mean_train = np.mean(data_train['Age'])
#经类别型的one-hot编码
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')

data_train = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
#删除无用的列
data_train.drop(['PassengerId','Pclass','Name','Sex','Ticket','Cabin','Embarked'],axis=1, inplace=True)
#Age/Fare做标准化
from sklearn.preprocessing import StandardScaler
stdd = StandardScaler()

age_scale_param =stdd.fit(data_train['Age'].values.reshape(-1,1))

data_train['Age_scaled'] = stdd.fit_transform(data_train['Age'].values.reshape(-1,1),age_scale_param)

fale_scale_param = stdd.fit(data_train['Fare'].values.reshape(-1,1))

data_train['Fare_scaled'] = stdd.fit_transform(data_train['Fare'].values.reshape(-1,1),fale_scale_param)
#删除无用的列
data_train.drop(['Age','Fare'],axis=1, inplace=True)
data_train.head(10)
X_train = data_train.iloc[:,1:]
Y_train = data_train.iloc[:,0]

#跑模型
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
model.fit(X_train,Y_train)
data_test = pd.read_csv('../input/test.csv')
PassengerID = data_test['PassengerId']
data_test.loc[data_test['Fare'].isnull(),'Fare'] = 0

#填充缺失值
data_test.loc[data_test['Age'].isnull(),'Age'] = Age_mean_train

data_test.loc[data_test['Cabin'].notnull(),'Cabin'] = 'Yes'
data_test.loc[data_test['Cabin'].isnull(),'Cabin'] = 'No'

#将类别型变量one-hot编码
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')

data_test = pd.concat([data_test,dummies_Pclass,dummies_Sex,dummies_Cabin,dummies_Embarked],axis=1)

data_test['Age_scale'] = stdd.fit_transform(data_test['Age'].values.reshape(-1,1),age_scale_param)

data_test['Fare_scale'] = stdd.fit_transform(data_test['Fare'].values.reshape(-1,1),fale_scale_param)

data_test.drop(['PassengerId','Pclass','Name','Sex','Age','Fare','Cabin','Embarked','Ticket'],axis=1,inplace=True)
#利用模型进行分类
predictions = model.predict(data_test)
result = pd.DataFrame({'PassengerId':PassengerID.as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv("gender_submission.csv", index=False)