%matplotlib inline
%reload_ext autoreload
%autoreload 2
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import re
from collections import Counter
sns.set_style('whitegrid')
sns.set_palette('Set1')
datadir = '../input/'
# read data from file
train_data = pd.read_csv(f'{datadir}train.csv')
test_data = pd.read_csv(f'{datadir}test.csv')
train_data.head()
test_data.head()
train_data.describe()
test_data.describe()
train_data.info()
test_data.info()
def detect_outliers(Q1, Q3, df, col):
    outlier_indices = []
    IQR = Q3 - Q1
    outlier_step = 1.5 * IQR
    outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
    outlier_indices.extend(outlier_list_col)     
    return outlier_indices
train_data.Age.describe()
train_data.Fare.describe()
train_data.SibSp.describe()
train_data.Parch.describe()
train_data.Fare.plot(kind='kde')
train_data.Age.plot(kind='kde')
out1 = detect_outliers(20.125, 38, train_data, 'Age')
out2 = detect_outliers(7.9, 31, train_data, 'Fare')
np.array(out1)
np.array(out2)
train_data.loc[out2]
# 备份数据
# 因为要做一些统一处理，所以将训练集测试集拼接起来
train = train_data
test = test_data
all_data = pd.concat([train, test], axis=0, sort=False)
train.shape, test.shape, all_data.shape
# one-hot编码
def dummies(col,data):
    data_dum = pd.get_dummies(data[col])
    data = pd.concat([data, data_dum], axis=1)
    data.drop(col, axis=1, inplace=True)
    return data
# 获取所有的列名
all_data.columns.tolist()
train.Pclass.value_counts()
sns.countplot(y='Pclass', hue='Survived', data=all_data)
any(all_data.Pclass.isna())
all_data.Pclass.head()
all_data = dummies('Pclass', all_data)
all_data.head()
all_data.rename(columns={1: 'Pclass1', 2: 'Pclass2', 3: 'Pclass3'}, inplace=True)
all_data.head()
backup1 = all_data
train = all_data[:891]
test = all_data[-418:]
# 恢复数据
all_data = backup1
any(all_data.Name.isna())
all_data.Name.head(10)
title = all_data.Name.map(lambda x: re.compile(",(.*?)\.").findall(x)[0].strip())
title.unique()
# 将 title 添加到最后
all_data['Title'] = title
all_data.drop('Name', axis=1, inplace=True)
all_data.head()
backup2 = all_data
train = all_data[:891]
test = all_data[-418:]
sns.countplot(y='Title', hue='Survived', data=all_data)
all_data.Title.value_counts()
all_data.Title.replace({'Mlle': 'Other', 'Lady': 'Other', 'Dona': 'Other', 'Jonkheer': 'Other', 'Mme': 'Other', 
                        'Don': 'Other', 'Sir': 'Other', 'Rev': 'Other', 'Col': 'Other', 'Major': 'High', 'Ms': 'Other',
                        'Master': 'High', 'Dr': 'High', 'the Countess': 'High', 'Capt': 'High'}, inplace=True)
all_data.Title.unique()
all_data.Title.value_counts()
sns.countplot(y='Title', hue='Survived', data=all_data)
all_data = dummies('Title', all_data)
backup2 = all_data
all_data = backup2
any(all_data.Sex.isna())
all_data.Sex.value_counts()
sns.countplot(y='Sex', hue='Survived', data=all_data)
all_data = dummies('Sex', all_data)
all_data.head()
backup3 = all_data
all_data = backup3
any(all_data.Age.isna()) # 存在空值
num_nona = all_data.Age.count() # 非空数值个数
num_nona
num_na = all_data.Age.shape[0] - num_nona  # 空值个数
num_na
# 去掉空值行
nona_data = all_data.Age.dropna(axis=0, how='all', inplace=False)
# 未缺失的年龄的大致分布
nona_data.plot(kind='kde')
nona_data.describe()
data = all_data.values
data.shape
len = data.shape[0]
for idx in range(len):
    tmp = data[idx, 2]
    if np.isnan(tmp):
        if data[idx, 12] == 1:
            data[idx, 2] = 39.
#         elif data[idx, 13] == 1 or data[idx, 14] == 1:
#             data[idx, 2] = 28.
        else:
            data[idx, 2] = 28.
    else:
        continue
all_data['Age'] = data[:, 2]
all_data.head()
all_data.shape
any(all_data.Age.isna())
# 整体年龄的大致分布
all_data.Age.plot(kind='kde')
# 未缺失的年龄的大致分布
nona_data.plot(kind='kde')
all_data.drop(columns=['Ticket'], axis=1, inplace=True)
all_data.head()
all_data.shape
nona = all_data.Cabin.count()
nona
num_na =all_data.shape[0] - nona
num_na
all_data.drop(columns=['Cabin'], axis=1, inplace=True)
all_data.head()
all_data.shape
any(all_data.Embarked.isna())
nona = all_data.Embarked.count()
nona
num_na =all_data.shape[0] - nona
num_na
all_data.Embarked.describe()
all_data.Embarked.fillna('S', inplace=True)
all_data.head()
any(all_data.Embarked.isna())
all_data = dummies('Embarked', all_data)
all_data.head()
any(all_data.SibSp.isna())
any(all_data.Parch.isna())
all_data['Family'] = all_data['SibSp'] + all_data['Parch']
all_data.head()
all_data.drop(columns=['SibSp', 'Parch'], axis=1, inplace=True)
all_data.head(10)
any(all_data.Fare.isna())
nona = all_data.Fare.count()
nona
num_na =all_data.shape[0] - nona
num_na
all_data.Fare.describe()
all_data.Fare.fillna(15, inplace=True)
any(all_data.Fare.isna())
all_data.head()
all_data.PassengerId.dtype
all_data.Survived.dtype
all_data.Age.dtype
all_data.Fare.dtype
all_data.Pclass1.dtype
all_data.Pclass2.dtype
all_data.Pclass3.dtype
all_data.High.dtype
all_data.Miss.dtype
all_data.Mr.dtype
all_data.Mrs.dtype
all_data.female.dtype
all_data.male.dtype
all_data.C.dtype
all_data.Q.dtype
all_data.S.dtype
all_data.Family.dtype
all_data.head()
all_data.Age = all_data.Age.apply(pd.to_numeric).astype('float32')
all_data.head()
all_data.PassengerId.dtype
all_data.Survived.dtype
all_data.Age.dtype
all_data.Fare.dtype
all_data.Pclass1.dtype
all_data.Pclass2.dtype
all_data.Pclass3.dtype
all_data.High.dtype
all_data.Miss.dtype
all_data.Mr.dtype
all_data.Mrs.dtype
all_data.female.dtype
all_data.male.dtype
all_data.C.dtype
all_data.Q.dtype
all_data.S.dtype
all_data.Family.dtype
train = all_data[:891]
test = all_data[-418:]
test = test.drop(columns=['Survived'], axis=1, inplace=False)
# train.to_csv(f'{datadir}train_process.csv', index=False)
# test.to_csv(f'{datadir}test_process.csv', index=False)
all_data.head()
part_data = all_data.drop(columns=['High', 'Miss', 'Mr', 'Mrs', 'Other'], axis=1, inplace=False)
part_data.head()
train1 = part_data[:891]
test1 = part_data[-418:]
test1 = test1.drop(columns=['Survived'], axis=1, inplace=False)
# train1.to_csv(f'{datadir}part_train_process.csv', index=False)
# test1.to_csv(f'{datadir}part_test_process.csv', index=False)
