# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import missingno as msno

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')



#看一下表格的基本结构

train_df.head()
# 二话不说，先来一个数据去重

train_df.duplicated(keep='first')

train_df.duplicated(keep='first')
#看一下数据的形状

print(train_df.shape)

print(test_df.shape)
#数据的字段类型也看一下

train_df.info()
#看一下是否存在数据缺失

print(train_df.isnull().any())

print(test_df.isnull().any())
#使用missingno 模块可以很方便的分析数据

msno.matrix(train_df)
msno.matrix(test_df)
'''

可以看到Age, Cabin 字段存在缺失，Cabin字段先不做处理，因为我也不知道客舱的编号是怎么排布的

修补 Age的时候使用平均值

'''





def nan_padding(data, columns):

    impute = SimpleImputer(missing_values=np.NaN, strategy='mean')

    for column in columns:

        data[column] = impute.fit_transform(data[column].values.reshape(-1, 1))

    return data



nan_columns = ['Age']

train_df = nan_padding(train_df, nan_columns)

test_df = nan_padding(test_df, nan_columns)



#检查一下是否修复好了

print(train_df.isnull().any())

print(test_df.isnull().any())
'''

移除 'PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin', 'Embarked' 字段， 这样做有点唐突， 因为我还没验证 这些字段和 Survived的相关性

但是简单处理一下

'''



def drop_not_concerned(data, columns):

    return data.drop(columns, axis=1)





drop_columns = ['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin', 'Embarked']

train_df = drop_not_concerned(train_df, drop_columns)

test_df = drop_not_concerned(test_df, drop_columns)



#看一下现在的数据

print(train_df)

print(test_df)
#看一下Pcalss ，Age， Parch ，  SibSp 的分布, 单身狗比较多



def features_hist(data, columns):

    for column in columns:

        data[column].hist()

        plt.show()

        

    

features_columns = ['Pclass', 'Age', 'Parch', 'SibSp']

features_hist(train_df, features_columns)
# 对 Pclass 字段进行 one_hot 编码

def dummy_data(data, columns):

    for column in columns:

        data = pd.concat([data, pd.get_dummies(data[column], prefix=column)], axis=1)

        data = data.drop(column, axis=1)

    return data





dummy_columns = ['Pclass']

train_df = dummy_data(train_df, dummy_columns)

test_df = dummy_data(test_df, dummy_columns)

train_df
#对Sex字段#进行处理

def sex_to_int(data):

    le = LabelEncoder()

    le.fit(['male', 'female'])

    data['Sex'] = le.transform(data['Sex'])

    return data

    



train_df = sex_to_int(train_df)

test_df = sex_to_int(test_df)

train_df
# Age 字段属于有界 离散数据，进行归一化处理

def normalize_age(data):

    scaler = MinMaxScaler()

    data['Age'] = scaler.fit_transform(data['Age'].values.reshape(-1, 1))

    return data



train_df = normalize_age(train_df)

test_df = normalize_age(test_df)

train_df
#分割训练数据和验证数据

def split_valid_test_data(data, fraction=(1 - 0.8)):

    data_y = data['Survived']

    lb = LabelBinarizer()

    data_y = lb.fit_transform(data_y)



    data_x = data.drop(['Survived'], axis=1)

    return train_test_split(data_x, data_y, test_size=fraction)





train_x, test_x, train_y, test_y = split_valid_test_data(train_df)