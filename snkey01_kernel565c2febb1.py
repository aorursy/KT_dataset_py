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
data = pd.read_csv("../input/train.csv")

data.head()
# 缩小数据范围

data = data.query("x>1.0 & x<1.25 & y>2.5 & y<2.75")

print(data.shape)
# 将时间戳转换为年月日时分秒

time = pd.to_datetime(data['time'], unit='s')

# 把日期转换为字典数据

time = pd.DatetimeIndex(time)
# 增加特征

data['day'] = time.day

data['hour'] = time.hour

data['weekday'] = time.weekday
# 删除特征

data = data.drop(['time'], axis=1)
# 查看数据

data.head()
# 将签到位置比较少的数据删掉

# 统计次数

result = data.groupby('place_id').count()

# 去掉次数少于3的数据

result = result[result.row_id > 3]

# place_id变为一列

result = result.reset_index()

# 对原始数据进行删选

data = data[data['place_id'].isin(result.place_id)]
# 取出数据中的特征值和目标值

y = data['place_id']

x = data.drop(['place_id', 'row_id'], axis=1)

x.head()
from sklearn.model_selection import train_test_split

# 进行数据的分割，训练集和测试集

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
from sklearn.preprocessing import StandardScaler



# 数据标准化, 将数据特征聚集在均值为0，方差为1的范围内

std = StandardScaler()

x_train = std.fit_transform(x_train)

x_test = std.fit_transform(x_test)
from sklearn.neighbors import KNeighborsClassifier



# knn

knn = KNeighborsClassifier(n_neighbors=5)

# 输入数据

knn.fit(x_train, y_train)



# 预测值

y_predict = knn.predict(x_test)

#print("预测的目标签到位置为：", y_predict)



# 使用测试数据集，计算准确率

accuracy = knn.score(x_test, y_test)

print("预测的准确率:", accuracy)