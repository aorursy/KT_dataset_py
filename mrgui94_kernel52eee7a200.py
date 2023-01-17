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
# 导入相关包

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
df_train.head(5)
#返回每列列名,该列非nan值个数,以及该列类型

df_train.info()
#返回数值型变量的统计量

df_train.describe()
#存活人数

df_train['Survived'].value_counts()
#相关性协方差表,corr()函数,返回结果接近0说明无相关性,大于0说明是正相关,小于0是负相关.

train_corr = df_train.drop('PassengerId',axis=1).corr()

train_corr
#画出相关性热力图

a = plt.subplots(figsize=(15,9))  # 调整画布大小

a = sns.heatmap(train_corr, vmin=-1, vmax=1 , annot=True , square=True)  # 画热力图
df_train[['Pclass','Survived']].groupby(['Pclass']).mean()
df_train[['Pclass','Survived']].groupby(['Pclass']).mean().plot.bar()
#先将数据集合并,一起做特征工程(注意,标准化的时候需要分开处理)

#先将test补齐,然后通过pd.apped()合并

df_test['Survived'] = 0

train_test = df_train.append(df_test)
train_test = pd.get_dummies(train_test,columns=['Pclass'])

train_test.head()
train_test['Sex'] = pd.factorize(train_test['Sex'])[0]

train_test.head()
train_test['SibSp_Parch'] = train_test['SibSp'] + train_test['Parch']

train_test = train_test.drop(['SibSp', 'Parch'], axis=1)

# pd.get_dummies(df_train,columns = ['SibSp','Parch','SibSp_Parch'])

train_test.head()
train_test = pd.get_dummies(train_test,columns=["Embarked"])

train_test.head()
#票价与pclass和Embarked有关，若有缺失值，用train分组后的平均数填充

# df_train["Pclass","Embarked", "Fare"].groupby(["Pclass","Embarked"]).mean()
# 用年龄是否缺失值来构造新特征

train_test.loc[train_test["Age"].isnull() ,"age_nan"] = 1

train_test.loc[train_test["Age"].notnull() ,"age_nan"] = 0

train_test = pd.get_dummies(train_test,columns=['age_nan'])

train_test.head()
train_test = train_test.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1)

train_test.head()
train_test = train_test.fillna(0)
train_data = train_test[:891]

test_data = train_test[891:]



train_data_X = train_data.drop(['Survived'],axis=1)

train_data_Y = train_data['Survived']

test_data_X = test_data.drop(['Survived'],axis=1)
from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(n_estimators=150,min_samples_leaf=2,max_depth=6,oob_score=True)

clf.fit(train_data_X,train_data_Y)

clf.oob_score_



df_test["Survived"] = clf.predict(test_data_X)

result = df_test[['PassengerId','Survived']].set_index('PassengerId')

result.to_csv('result1.csv')