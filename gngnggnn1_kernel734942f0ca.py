import pandas as pd #数据分析
import numpy as np #科学计算
from pandas import Series,DataFrame

data_train = pd.read_csv(r'../input/train.csv') 
data_train.info()
data_train.describe()
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['font.family']='sans-serif' 
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
fig = plt.figure()
fig.set(alpha=0.2) # 设定图表颜色alpha参数

plt.subplot2grid((2,3),(0,0)) # 在一张大图里分列几个小图
data_train.Survived.value_counts().plot(kind='bar')# 柱状图
plt.title(u"获救情况 (1为获救)") # 标题
plt.ylabel(u"人数") # Y轴标签

plt.subplot2grid((2,3),(0,1))
data_train.Pclass.value_counts().plot(kind="bar") # 柱状图显示
plt.ylabel(u"人数")
plt.title(u"乘客等级分布")

plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived, data_train.Age) #为散点图传入数据
plt.ylabel(u"年龄") # 设定纵坐标名称
plt.grid(b=True, which='major', axis='y')
plt.title(u"按年龄看获救分布 (1为获救)")

plt.subplot2grid((2,3),(1,0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde') # 密度图
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel(u"年龄")# plots an axis lable
plt.ylabel(u"密度")
plt.title(u"各等级的乘客年龄分布")
plt.legend((u'头等舱', u'2等舱',u'3等舱'),loc='best') # sets our legend for our graph.

plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title(u"各登船口岸上船人数")
plt.ylabel(u"人数")
plt.show()
Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts() # 未获救
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts() # 获救
df = pd.DataFrame({u'获救':Survived_1,u'未获救':Survived_0})
df.plot(kind = 'bar', stacked = True)
plt.title(u'各乘客等级的获救情况')
plt.xlabel(u'乘客等级')
plt.ylabel(u'人数')
plt.show()
Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
df = pd.DataFrame({u'男性':Survived_m,u'女性':Survived_f})
df.plot(kind = 'bar', stacked = True)
plt.title(u'按性别看获救情况')
plt.xlabel(u'性别')
plt.ylabel(u'人数')
plt.show()
fig = plt.figure()
plt.title(u'根据舱等级和性别的获救情况')

ax1 = fig.add_subplot(141) # 将图像分为1行4列，从左到右从上到下的第1块
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind = 'bar', label = 'female high class', color = '#FA2479')
ax1.set_xticklabels([u'获救',u'未获救'], rotation = 0) # 根据实际填写标签
ax1.legend([u'女性/高级舱'], loc = 'best')

ax2 = fig.add_subplot(142, sharey = ax1) # 将图像分为1行4列，从左到右从上到下的第2块
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind = 'bar', label = 'female low class', color = 'pink')
ax2.set_xticklabels([u"未获救", u"获救"], rotation=0)
plt.legend([u"女性/低级舱"], loc='best')

ax3 = fig.add_subplot(143, sharey = ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind = 'bar', label = 'male high class', color = 'lightblue')
ax3.set_xticklabels([u'未获救',u'获救'], rotation = 0)
plt.legend([u'男性/高级舱'], loc = 'best')

ax4 = fig.add_subplot(144, sharey = ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind = 'bar', label = 'male low class', color = 'steelblue')
ax4.set_xticklabels([u'未获救',u'获救'], rotation = 0)
plt.legend([u'男性/低级舱'], loc = 'best')
plt.show()
fig = plt.figure()
fig.set(alpha = 0.2)
Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
df = pd.DataFrame({u'获救':Survived_1,u'未获救':Survived_0})
df.plot(kind = 'bar', stacked = True)
plt.title(u'各登陆港口乘客的获救情况')
plt.xlabel(u'登陆港口')
plt.ylabel(u'人数')
plt.show()
fig = plt.figure()
fig.set(alpha = 0.2)

Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
df = pd.DataFrame({u'有':Survived_cabin, u'无':Survived_nocabin}).transpose()
df.plot(kind = 'bar', stacked = True)
plt.title(u'按Cabin有无看获救情况')
plt.xlabel(u'Cabin有无')
plt.ylabel(u'人数')
plt.show()
#我们将测试集导入，再将删除Survived数据的训练集与测验集进行合并，这样便于进行数据处理
data_test = pd.read_csv(r'../input/test.csv') # 导入测验集数据

y = data_train['Survived'] # 将训练集Survived 数据存储在y中
del data_train['Survived'] # 删除训练集Survived数据

sum_id = data_test['PassengerId'] # 存储测试集乘客ID

df = pd.merge(data_train, data_test,how='outer') # 合并无Survived数据的训练集与测验集，how = ‘outer’ 意为并集

#删掉无关因素
df = df.drop(['Name','PassengerId','Ticket','Cabin'],axis=1) # 删除姓名、ID、船票信息、客舱信息，axis=0 删除行，=1 删除列

#缺失数据填充
df['Age'] = df['Age'].fillna(df['Age'].mean())  # 用平均值填充空值
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
df['Embarked'] = df['Embarked'].fillna( df['Embarked'].value_counts().index[0]) # 用数量最多项填充

#将性别与港口用哑变量表示
dumm = pd.get_dummies(df[['Sex','Embarked']]) # '哑变量'矩阵
df = df.join(dumm)
del df['Sex'] # 删除
del df['Embarked']

#数据降维
df['Age'] = (df['Age']-df['Age'].min()) /(df['Age'].max()-df['Age'].min())

df['Fare'] = (df['Fare']-df['Fare'].min()) /(df['Fare'].max()-df['Fare'].min())

#训练模型
data_train = df[:len(data_train)] # 将合并后的数据分离
data_test = df[len(data_train):]
from sklearn.model_selection import train_test_split

X_train, X_val, y_train,y_val = train_test_split(data_train,y,test_size=0.3, random_state=52) # 以7：3（0.3）将训练集与获救结果随机拆分，随机种子为42

from sklearn.linear_model import LogisticRegression # 引入逻辑回归
LR = LogisticRegression()

LR.fit(X_train, y_train) # 训练数据
print('训练集准确率：\n',LR.score(X_train, y_train)) # 分数
print('验证集准确率：\n',LR.score(X_val, y_val))

#预测测验集
pred= LR.predict(data_test) # pred 为预测结果

pred = pd.DataFrame({'PassengerId':sum_id.values, 'Survived':pred}) # 格式化预测结果

pred.to_csv('pred_LR.csv',index=None) # 导出数据
