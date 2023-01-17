# 第三方库

import numpy as np

import pandas as pd

import os

import gc

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')
# 数据输入

data1 = pd.read_csv('../input/netflix-prize-data/combined_data_1.txt', header=None, names=['custID', 'rating'], usecols=[0, 1], dtype={'custID': 'str', 'rating': 'str'});

print('file1 is loaded!');

data2 = pd.read_csv('../input/netflix-prize-data/combined_data_1.txt', header=None, names=['custID', 'rating'], usecols=[0, 1], dtype={'custID': 'str', 'rating': 'str'});

print('file2 is loaded!');

data3 = pd.read_csv('../input/netflix-prize-data/combined_data_1.txt', header=None, names=['custID', 'rating'], usecols=[0, 1], dtype={'custID': 'str', 'rating': 'str'});

print('file3 is loaded!');

data4 = pd.read_csv('../input/netflix-prize-data/combined_data_1.txt', header=None, names=['custID', 'rating'], usecols=[0, 1], dtype={'custID': 'str', 'rating': 'str'});

print('file4 is loaded!');

# 数据拼接

data = pd.concat([data1, data2, data3, data4], axis=0);

data.index = range(len(data));

print('四份数据已经拼接完毕！')

print(data.head());
# 释放多余内存

del data1, data2, data3, data4;

gc.collect();

print('内存释放完毕！')
# 电影ID所在的行索引

movie_index = list(data[pd.isnull(data['rating'])].index);

movie_index.append(len(data));

print('将电影ID所在行索引找出来并存储进movie_index，数据前五行：');

print(movie_index[:5]);



# 随机选取的行索引，选取其中的10%的数据

index_selected = pd.DataFrame(data[pd.notnull(data['rating'])].index).sample(frac=0.01, replace=False, axis=0);

#print(index_selected.head());

# 从小到大排序

index_selected = index_selected.sort_values(by=0, ascending=True);

print('随机选取的行索引（前五行）：')

print(index_selected.head());
# 随机选取的行索引下对应的电影ID

movie_np = [];

movie_id = 1;

for ind in index_selected[0].values:

    if ind > movie_index[movie_id]:

        movie_id += 1;

    movie_np.append(movie_id);

    

print('随机选取的数据集对应的电影ID（前五个数据）：');

print(movie_np[: 5]);
# 作出随机选取的数据

data_shorten = data.loc[index_selected[0].values];

data_shorten['movieID'] = movie_np;

data_shorten = data_shorten[['custID', 'movieID', 'rating']];

data_shorten.index = range(len(data_shorten));

print('随机挑选的数据（包括 custID, MovieID, rating）（前五行）为')

print(data_shorten.head());
# 查看 data_shorten 数据的数据类型

print(data_shorten.info());
# 将 custID 和 rating 的数据类型转换为 int 

data_shorten['custID'] = data_shorten['custID'].map(int);

data_shorten['rating'] = data_shorten['rating'].map(int);

print('数据类型转换成功！');

print(data_shorten.info());
# 释放data内存

del data, movie_index, movie_np, index_selected;

gc.collect();

print('释放内存成功！')
from surprise import BaselineOnly

from surprise import Dataset

from surprise import accuracy

from surprise import Reader

from surprise.model_selection import KFold
# 数据加载

reader = Reader();

data = Dataset.load_from_df(data_shorten, reader);
# 交叉验证

kf = KFold(n_splits=3);

algo = BaselineOnly();

for trainset, testset in kf.split(data):

    # 训练

    algo.fit(trainset);

    # 测试

    predictions = algo.test(testset);

    # 计算rmse误差

    accuracy.rmse(predictions, verbose=True);
# 读取probe集

probeData = pd.read_csv('../input/netflix-prize-data/probe.txt', header=None, names=['custID'], dtype={'custID': 'str'});

print('读取probe数据集成功！\n')

print('probe数据集前五行为')

print(probeData.head());
# 取出电影ID

movieID_index = []; # 存储电影ID所在行索引

movieID_np = []; # 存储custID对应的电影ID

movie_id = -1;

for i in range(len(probeData)):

    if ':' in probeData.custID[i]:

        movieID_index.append(i); 

        movie_id = int(probeData.custID[i][:-1]);

    else:

        movieID_np.append(movie_id);

              

# 电影ID的行索引

print('电影ID的行索引（前五个数据）为')

print(movieID_index[:5]);
# 处理probeData数据集

print('开始处理probe数据集！');

# 第一步：丢掉电影ID所在行

print('第一步：丢掉电影ID所在行。执行成功！');

probeData.drop(index=movieID_index, inplace=True);

# 第二步：加入电影ID列

print('第二步：加入电影ID列。执行成功！');

probeData['movieID'] = movieID_np;

# 第三步：index重排

print('第三步：index重排。执行成功！');

probeData.index = range(len(probeData));

# 第四步：数据类型转换（从object变到int）

print('第四步：数据类型转换。执行成功！');

probeData['custID'] = probeData['custID'].map(int);

probeData['movieID'] = probeData['movieID'].map(int);

print('处理probe数据完毕！')
# 数据集

print(probeData.head());

print(probeData.info());
# 类似的，我们选取probe数据集中的1w条数据作为新的probe数据集

probeData_shorten = probeData.sample(n=10000, replace=False, axis=0);

print(probeData_shorten.head());
# 求出probe数据集中每一对(custID, movieID)对应的rating 

print('开始遍历！');

porbe_rating = [];

for i in range(len(probeData_shorten)):

    temp = data_shorten['rating'][(data_shorten.custID==probeData.custID[i]) & (data_shorten.movieID==probeData.movieID[i])];

    rating = temp.values[0] if len(temp) else np.NaN;

    porbe_rating.append(rating);

    #print(i);

print('遍历结束！');

print('probe数据集(custID, movieID)对应的rating列（前五行）：');

print(porbe_rating[:5]);
# 新的probeData_shorten数据集

probeData_shorten['rating'] = porbe_rating;

new_probeData_shorten = probeData_shorten[pd.notnull(probeData_shorten['rating'])];



print('带有评分的probe数据集（前五行）为');

print(new_probeData_shorten.head());
# 数据加载

reader = Reader();

probe_data = Dataset.load_from_df(new_probeData_shorten, reader);



# 交叉验证

kf = KFold(n_splits=3);

algo = BaselineOnly();

for trainset, testset in kf.split(probe_data):

    # 训练

    algo.fit(trainset);

    # 测试

    predictions = algo.test(testset);

    # 计算rmse误差

    accuracy.rmse(predictions, verbose=True);