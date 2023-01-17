# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra 科学运算

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv) 数据分析



data_train = pd.read_csv('../input/titanic/train.csv')

data_train
data_train.info()
# 可以看到数值型的列的信息, 对于文本型不起作用

data_train.describe()
import matplotlib.pyplot as plt # 引入绘图模块

fig = plt.figure(figsize=(20,12)) # 创建一个窗口

fig.set(alpha=0.2) # 不知道干嘛用的



# 观察幸存者和遇难者的数量

plt.subplot2grid((2,3),(0,0)) # 划分子图, 2行3列, 在第0行第0列画, grid(格子)

data_train['Survived'].value_counts().plot(kind='bar') # values_counts 查看不同的值及其数量 plot() 转换成图表

plt.title("Survived (1 for Yes)")

plt.ylabel('NB of people')



# 观察幸存者的舱位分布

plt.subplot2grid((2,3),(0,1))

data_train['Pclass'].value_counts().plot(kind='bar')

plt.title("Distribution of Ticket Class")

plt.ylabel('Nb of people')



# 观察幸存者的年龄分布

plt.subplot2grid((2,3),(0,2))

plt.scatter(data_train['Survived'], data_train['Age'])

plt.ylabel('Age')

plt.title('Ditribution of Survived by Age')



# 不同年龄的人再不同舱位上的密度分布

plt.subplot2grid((2,3),(1,0),colspan=2)

data_train['Age'][data_train['Pclass']==1].plot(kind='kde') # KDE Kernel Density Estimation 核密度估计

data_train['Age'][data_train['Pclass']==2].plot(kind='kde')

data_train['Age'][data_train['Pclass']==3].plot(kind='kde')

plt.xlabel('Age')

plt.title('Distribution of Age by Tiket class')

plt.legend(['Class 1', 'Class 2', 'Class 3'], loc='best') # legend: 图例, best 代表自动寻找最好的位置摆放标签



# 从不同港口登船的人数分布

plt.subplot2grid((2,3),(1,2))

data_train['Embarked'].value_counts().plot(kind='bar')

plt.ylabel('NB of people')

plt.title('Distribution of People by Port Embarked')



# 调整fugure大小可以一定程度上避免子图重叠

plt.show()
# 观察舱位与幸存状况的关系

Survived_0 = data_train['Pclass'][data_train['Survived']==0].value_counts()

Survived_1 = data_train['Pclass'][data_train['Survived']==1].value_counts()

df = pd.DataFrame({'Yes':Survived_1, 'No':Survived_0})

df.plot(kind='bar')

plt.title("Situation of Rescue by Passenger Class")

plt.xlabel(u"Passenger Class") 

plt.ylabel(u"Nb of people") 

plt.show()
# 观察性别与幸存状况的关系

Survived_m = data_train['Survived'][data_train['Sex']=='male'].value_counts()

Survived_f = data_train['Survived'][data_train['Sex']=='female'].value_counts()

df = pd.DataFrame({'Male':Survived_m,'Female':Survived_f})

df.plot(kind='bar')

plt.title('Situation of Rescue by gendre')

plt.ylabel('Nb of people')

plt.xlabel('Gendre')

plt.show()
# 更详细的观察性别、舱位与幸存状况的关系

fig, axs = plt.subplots(2,2, figsize=(10,8), sharey=True)

fig.suptitle('Distribution of Survived by Gendre and Class leavel\n(0: No, 1: Yes)')



# 中高级舱位的女性幸存情况

female_Premium_Class = data_train.Survived[data_train['Sex'] == 'female'][data_train['Pclass'] != 3].value_counts() # 选取1和2等舱的女性乘客

female_Premium_Class.plot(kind='bar', ax=axs[0][0], color='red')

axs[0][0].set_title('Female in Senior/Inter Class')

axs[0][0].set_xticklabels(female_Premium_Class.index, rotation=0)

axs[0][0].set_ylabel('Nb of people')



# 经济舱的女性幸存情况

female_Economy_Class = data_train.Survived[data_train['Sex'] == 'female'][data_train['Pclass'] == 3].value_counts() # 选取3等舱的女性乘客

female_Economy_Class.plot(kind='bar', ax=axs[0][1], color='pink')

axs[0][1].set_title('Female in Eco Class')

axs[0][1].set_xticklabels(female_Economy_Class.index, rotation=0)

axs[0][0].set_ylabel('Nb of people')



# 中高级舱位的男性幸存情况

male_Premium_Class = data_train.Survived[data_train['Sex'] == 'male'][data_train['Pclass'] != 3].value_counts() # 选取1和2等舱的男性乘客

male_Premium_Class.plot(kind='bar', ax=axs[1][0], color='lightblue')

axs[1][0].set_title('Male in Senior/Inter Class')

axs[1][0].set_xticklabels(male_Premium_Class.index, rotation=0)

axs[0][0].set_ylabel('Nb of people')



# 经济舱的男性幸存情况

male_Economy_Class = data_train.Survived[data_train['Sex'] == 'male'][data_train['Pclass'] == 3].value_counts() # 选取3等舱的男性乘客

male_Economy_Class.plot(kind='bar', ax=axs[1][1], color='steelblue')

axs[1][1].set_title('Male in Eco Class')

axs[1][1].set_xticklabels(male_Economy_Class.index, rotation=0)

axs[0][0].set_ylabel('Nb of people')



plt.show()
# 从各港口登船的人的幸存状况

Survived_0 = data_train.Embarked[data_train['Survived']==0].value_counts()

Survived_1 = data_train.Embarked[data_train['Survived']==1].value_counts()

df = pd.DataFrame({'Yes':Survived_1, 'No':Survived_0})

df.plot(kind='bar', stacked=True)

plt.show()
# 观察从兄妹数量(SibSp)幸存情况的关系

g = data_train.groupby(['SibSp', 'Survived'])

df = pd.DataFrame(g.count()['PassengerId'])

df
# 观察父母孩子(Parch)数量和幸存情况的关系

g = data_train.groupby(['Parch', 'Survived'])

df = pd.DataFrame(g.count()['PassengerId'])

df
# 最后一个特征 船票编号

data_train.Cabin.value_counts()
# 处理Cabin

data_train.loc[(data_train.Cabin.notnull()), 'Cabin'] = 'Yes'

data_train.loc[(data_train.Cabin.isnull()), 'Cabin'] = 'No'

data_train.Cabin


# 对类目型特征因子化

dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')

dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')

dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')

dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')

df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)

df.drop(['Age', 'Name', 'Pclass', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

df
# 数据归一化, Fare里面的值太大, 会降低收敛速度或者不收敛

import sklearn.preprocessing as preprocessing

scaler = preprocessing.StandardScaler()

df['Fare_scaled'] = scaler.fit_transform(df[['Fare']])

df.drop(['Fare'], axis=1, inplace=True)

df.iloc[:, 1:]
from sklearn import linear_model

train_np = df.iloc[:, 1:].values

y = train_np[:, 0]

x = train_np[:, 1:]
clf = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6)

clf.fit(x,y)



clf
# 处理test_data



data_test = pd.read_csv('../input/titanic/test.csv')

data_test.info()
# 测试数据中 Fare有一个是Nan， 改为0

data_test.loc[(data_test.Fare.isnull()), 'Fare'] = 0



data_test.loc[(data_test.Cabin.notnull()), 'Cabin'] = 'Yes'

data_test.loc[(data_test.Cabin.isnull()), 'Cabin'] = 'No'



dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix='Cabin')

dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix='Embarked')

dummies_Sex = pd.get_dummies(data_test['Sex'], prefix='Sex')

dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix='Pclass')

df = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)

df.drop(['Age', 'Name', 'Pclass', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

df['Fare_scaled'] = scaler.fit_transform(df[['Fare']])

df.drop(['Fare'], axis=1, inplace=True)

df
test = df.iloc[:, 1:].values

test
pred = clf.predict(test)

pred
result = pd.DataFrame({'PassengerId':data_test['PassengerId'], 'Survived':pred})

result.to_csv("logistic_regression_predictions.csv", index=False)

result