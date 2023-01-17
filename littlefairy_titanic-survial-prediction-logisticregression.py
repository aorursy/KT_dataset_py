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
# 导入训练数据集

train = pd.read_csv('../input/train.csv')

# 导入测试数据集

test = pd.read_csv('../input/test.csv')
# 查看数据

train.head()
test.head()
train.info()
# 查看数据大小

print('训练数据集：',train.shape,'测试数据集：',test.shape)
# 合并数据集，方便对两个数据集进行清洗

full = train.append(test,ignore_index=True)

print('合并后的数据集：',full.shape)
# 获取描述统计信息

full.describe()
# 查看数据缺失情况

full.info()
# 年龄（Age）缺失值填充

full['Age'] = full['Age'].fillna(full['Age'].mean())

# 船票价格（Fare）缺失值填充

full['Fare'] = full['Fare'].fillna(full['Age'].mean())
# 字符串类型特征处理

# 登船港口（Embarked）

'''

将缺失值填充为最频繁出现的值

'''

pd.value_counts(full['Embarked'])
full['Embarked'] = full['Embarked'].fillna('S')
# 船舱号（Cabin）

full['Cabin'].head()
'''

缺失值太多，船舱号缺失值填充为‘U’，表示未知

'''

full['Cabin'] = full['Cabin'].fillna('U')
# 查看缺失处理后的数据

full.info()
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

full['Name'].head()
# 分割姓和名,提取Mrs或Mr

# test

name1 = 'Braund, Mr. Owen Harris'

str1 = name1.split(',')[1]

print(str1)

# 继续拆分姓，拆分符号‘.’

str2 = str1.split('.')[0]

print(str2)
'''

定义函数：从姓名中提取称谓

'''

def extractTitle(name):

    str1 = name.split(',')[1]

    str2 = str1.split('.')[0]

#     删除空格

    str3 = str2.strip()

    return str3
# 1.存放提取后的特征

titleDf = pd.DataFrame()

# 2.使用map函数：对Seris的每个数据应用自定义的函数

titleDf['Title'] = full['Name'].map(extractTitle)

titleDf.head()
# 查看称谓类别

titleDf['Title'].value_counts()
# 设置itle和头衔的映射字典

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
# 存放客舱号信息

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
full.shape
# 生成相关系数矩阵

corrDf = full.corr()

corrDf
'''

查看各个特征与生存情况(Survived)的相关系数，

ascending = False 表示降序排列

'''

corrDf['Survived'].sort_values(ascending=False)
'''

选取正相关性或负相关性最高的几个特征，作为模型的输入特征：

头衔(titleDf)，客舱等级(pclassDf),家庭大小(familyDf),船票价格(Fare),船舱号(cabinDf),登船港口(embarkedDf),性别(Sex)

'''

full_X = pd.concat([titleDf,#头衔

                    pclassDf,#客舱等级

                    familyDf,#家庭大小

                    full['Fare'],#船票价格

                    cabinDf,#船舱号

                    embarkedDf,#登船港口

                    full['Sex']#性别

                   ],axis=1)

full_X.head()
# 原始数据集共有891行

sourceRow = 891

# 存储原始数据集：特征

source_X = full_X.loc[0:sourceRow-1,:]

# 存储原始数据集：标签

source_y = full.loc[0:sourceRow-1,'Survived']



# 预测数据集

pred_X = full_X.loc[sourceRow:,:]
'''

确保原始数据集大小为891，且与预测数据集维数相同，以防止构建模型时报错

'''

print('原始数据集大小：',source_X.shape)

print('预测数据集大小：',pred_X.shape)
'''

拆分训练数据和测试数据

'''

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(source_X,source_y,train_size=.8)

print('训练数据集特征:',X_train.shape,

     '测试数据集特征：',X_test.shape,

     )

print('训练数据集标签：',y_train.shape,

     '测试数据集标签：',y_test.shape)
source_y.head()
'''

生存预测问题为分类问题

选取逻辑回归算法进行模型训练

'''

# 1.导入算法

from sklearn.linear_model import LogisticRegression

# 2.创建模型

model = LogisticRegression()

# 3.训练模型

model.fit(X_train,y_train)
# 训练模型

model.fit(X_train,y_train)
# 模型评估

model.score(X_test,y_test)
# 使用机器学习模型，对预测数据集中的生存情况进行预测

pred_y = model.predict(pred_X)
pred_y
'''

预测结果为float类型，kaggle要求提交的结果为int型，

需要转换数据类型

'''

pred_y = pred_y.astype(int)
# 乘客id

passengerId = full.loc[sourceRow:,'PassengerId']

# 数据框：乘客id,预测生存情况

predDf = pd.DataFrame(

{'PassengerId':passengerId,

'Survived':pred_y})

predDf.shape
predDf.head()
# 保存结果

predDf.to_csv('titanic_pred.csv',index=False)