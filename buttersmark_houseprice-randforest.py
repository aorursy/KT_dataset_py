import numpy as np # 用Python进行科学计算的基础软件包。

import pandas as pd # Pandas是一个强大的分析结构化数据的工具集；它的使用基础是Numpy；用于数据挖掘和数据分析，同时也提供数据清洗功能。

from sklearn.model_selection import train_test_split #用于分割测试样本的函数

import warnings

warnings.filterwarnings('ignore') # 忽略警告消息

import os

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv') #读取训练集

data.head(5)
print("训练集的尺寸：",data.shape) # 训练集的尺寸

print(data.describe()) # 得到训练集的基本特征

print("是否有空值：",pd.isna(data).any().any()) # 检查空值情况
data.fillna(0, inplace = True) # inplace参数为True代表直接在表本身上修改，默认为False

data = data.drop('Id', axis = 1) # axis默认问0，0代表以行索引检索，1代表以列索引检索
data = pd.get_dummies(data) # 对离散值进行One-hot编码

data.head(5)
data.shape
X = data.drop('SalePrice', axis = 1) # axis默认问0，0代表以行索引检索，1代表以列索引检索

Y = np.array(data['SalePrice']) # 将dataframe转为array，array相较于list有更多便捷的函数
feature_list = list(X.columns) # 提取特征列表后面会用到

feature_list
X = np.array(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .5, random_state = 10) # test_size是测试数据集占比。random_state默认为None，代表生成随机的种子，若种子一样，每次抽取的结果都一样
print(X_train.shape,

X_test.shape,

Y_test.shape,

Y_train.shape)
from sklearn.ensemble import RandomForestRegressor #导入随机森林算法函数
rf = RandomForestRegressor(n_estimators = 2000, random_state=10) #n_estimators是生成的数(决策树)的个数，因为随机森林不会过拟合，所以树越多越准确



rf.fit(X_train, Y_train) #训练模型



predictions = rf.predict(X_test) #测试模型



errors = abs(predictions - Y_test) #将预测值数组与实际值数组做差求绝对值得到每个结果的误差



print('Mean Absolute Error:', round(np.mean(errors),2),'degrees.') #将误差求均值，平方，得到平均误差的平方
exam = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

container = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

lines = container.shape[0] #得到原训练集的行数

container = container.drop('SalePrice', axis = 1) #去除原训练集的结果列

IDs = exam[['Id']] #将测试集ID存起来

exam = pd.concat([container,exam]) # 上下合并两个表

exam.fillna(0, inplace = True)

exam = exam.drop('Id', axis = 1)

exam = pd.get_dummies(exam)

exam = exam.iloc[lines:,] # 去掉原训练集

exam = exam.loc[:,feature_list] # 根据原特征获取列，如果列与原训练集对不上将会在使用模型预测时报错

pd.isna(exam).any().any()

exam = np.array(exam)

results = rf.predict(exam)

results = pd.DataFrame(results) # 将results由array转为dataframe

results = results.rename(columns={0:'SalePrice'}) # 将列名改为SalePrice

output = IDs.join(results) # 合并ID和预测结果

output.to_csv('/natthew.csv',index = 0)# 输出csv，index默认为1，代表是否保留行索引