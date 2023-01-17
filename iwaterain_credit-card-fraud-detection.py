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
# 银行卡刷卡交易数据准备，常用 python package

import numpy as np # 数值计算

import pandas as pd # 数据集处理工具

import seaborn as sns # 可视化

import matplotlib.pyplot as plt # 画图工具

import matplotlib.gridspec as gridspec 



# 使得图片可以在 jupyter notebook 中呈现

%matplotlib inline
# Load data

df_credit = pd.read_csv("../input/creditcardfraud/creditcard.csv")
# 查看前5行数据

df_credit.head()
# 查看数据类型和是否有空值，发现数据集有28个变量，而且没有空值。

df_credit.info()
# 查看正常0和盗刷1分类统计个数：

print("统计分布：正常0 vs 盗刷1:")

print(df_credit['Class'].value_counts())
plt.figure(figsize=(7,5))

sns.countplot(df_credit['Class'])

plt.title('Distribution', fontsize=18)

plt.ylabel(" # of transaction", fontsize=15)

plt.show()
# 查看盗刷和正常交易之间的交易金额统计结果

df_fraud = df_credit[df_credit['Class'] == 1]

df_normal = df_credit[df_credit['Class'] == 0]



print('盗刷金额统计')

print(df_fraud["Amount"].describe())

print('正常交易金额统计')

print(df_normal["Amount"].describe())
columns = df_credit.iloc[:1:29].columns # 删除已探索过的变量



frauds = df_credit.Class == 1

normals = df_credit.Class == 0



grid = gridspec.GridSpec(14,2)

plt.figure(figsize=(15,20*4))



for n, col in enumerate(df_credit[columns]):

    ax = plt.subplot(grid[n])

    sns.distplot(df_credit[col][frauds], bins=50, color='g') # 绿色描绘盗刷样本分布

    sns.distplot(df_credit[col][normals], bins=50, color='r') # 红色描绘正常样本分布

    ax.set_ylabel('Density')

    ax.set_title(str(col))

    ax.set_xlabel('')

plt.show()
timedelta = pd.to_timedelta(df_credit['Time'], unit='s')

df_credit['Time_min']  = (timedelta.dt.components.minutes).astype(int)

df_credit['Time_hour'] = (timedelta.dt.components.hours).astype(int)



# 从小时上看，寻找insight

plt.figure(figsize=(12,5))

sns.distplot(df_credit[df_credit['Class'] ==0]["Time_hour"],color='g')

sns.distplot(df_credit[df_credit['Class'] ==1]["Time_hour"],color='r')



plt.title('Fraud X Normal Transactions by Hours', fontsize=17)

plt.xlim([-1,25])

plt.show()