# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np # 科学计算库

import pandas as pd # 导入数据文件

import seaborn as sns# 数据可视化

import matplotlib.pyplot as plt# 数据可视化库

import warnings# 忽略警告

import pandas_profiling as ppf# 数据eda

from sklearn.preprocessing import LabelEncoder# 标签编码

from sklearn.preprocessing import MinMaxScaler# 归一化

from sklearn.model_selection import train_test_split# 数据集划分

from sklearn.linear_model import LinearRegression# 算法

from sklearn.metrics import mean_absolute_error# 评估函数
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
train.head()# 显示数据文件前五行
train.head(10)# 显示数据文件前10行
train.info()# 看数据信息
test.info()
# 数据EDA

ppf.ProfileReport(train)
train["Age"] = train["Age"].fillna(np.mean(train["Age"]))# 填充年龄列的缺失值为平均值

test["Age"] = test["Age"].fillna(np.mean(test["Age"]))# 测试集同理

train.drop("Cabin",axis=1,inplace=True)# 删除cabin这一列然后再替代

test.drop("Cabin",axis=1,inplace=True)# 测试集同理
train.head(10)
train["Age"].isnull().sum()# 空值个数判断
train.drop("Name",axis=1,inplace=True)# 删除name这一列然后再替代

test.drop("Name",axis=1,inplace=True)# 测试集同理

train.drop("Ticket",axis=1,inplace=True)# 删除ticket这一列然后再替代

test.drop("Ticket",axis=1,inplace=True)# 测试集同理

train.drop("Embarked",axis=1,inplace=True)# 删除embarked这一列然后再替代

test.drop("Embarked",axis=1,inplace=True)# 测试集同理
train.head()
lab = LabelEncoder()# sklearn中都可以这么用

train["Sex"] = lab.fit_transform(train["Sex"])#编码：将字符串转化为数字

test["Sex"] = lab.fit_transform(test["Sex"])

train.head()
minmax = MinMaxScaler()

train["Age"] = minmax.fit_transform(np.array(train["Age"]).reshape(-1,1))# 归一化，大数字进行放大缩小

test["Age"] = minmax.fit_transform(np.array(test["Age"]).reshape(-1,1))

train["Age"].max()

train["Age"].min()
train["Fare"] = minmax.fit_transform(np.array(train["Fare"]).reshape(-1,1))# 归一化，大数字进行放大缩小

test["Fare"] = test["Fare"].fillna(np.mean(test["Fare"]))

test["Fare"] = minmax.fit_transform(np.array(test["Fare"]).reshape(-1,1))

test.info()
#划分训练集和测试集

x = train.drop("Survived",axis=1)

x.info()
y = train["Survived"]

y.shape
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=0)#划分训练集和测试集

x_train.shape,y_train.shape# 检查shape
# 使用sklearn 搭建模型训练

lin  = LinearRegression()# 实例化算法

lin.fit(x_train,y_train)# 对训练集进行拟合

y_pred = lin.predict(x_test)#对自己划分的测试集进行预测

mae = mean_absolute_error(y_pred,y_test)# 计算均值绝对误差

mae
predict = lin.predict(test)

predict# survive值
# 改成提交格式

submission = pd.DataFrame({'PassengerId':test["PassengerId"],'Survived':predict})# 以字典的形式建立dataframe

submission.to_csv("submmision_csv",index = False)# 转换为CSV文件

pd.read_csv("submmision_csv")