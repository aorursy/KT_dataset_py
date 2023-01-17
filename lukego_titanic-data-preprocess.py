import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

%matplotlib inline
data = pd.read_csv('../input/train.csv')
data.columns
data.shape
data.head()
data.info()
id_cols=['PassengerId','Ticket']

string_cols=['Name','Embarked','Cabin']

numeric_cols=['Age','SibSp','Parch','Fare']

discrete_cols=['Pclass','Sex']
counts = data['Survived'].value_counts()

print(counts)
print(counts[0]/counts[1])
# 查看数据中缺失值的比例

def naCount(data):

    naCount = {}

    for col in data.columns:

        a = data[col].isnull()

        b = data[col].apply(lambda x:True if len(str(x).strip())==0 else False)

        c = [i or j for i,j in zip(a,b)]

        colNa = sum(c)

        naCount[col] = [colNa, '{:0.2f}%'.format((float(colNa)/len(data))*100)]

    return naCount
naCount(data)
data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric,errors='coerce')

data[string_cols] = data[string_cols].astype(str)
# 数值类型特征箱线图

i = 0

plt.figure(figsize=(30,15))

for cols in data[numeric_cols]:

    plt.subplot(221+i)

    data[cols].plot.box()

    i+=1
# 数值类型特征条形图

i = 0

plt.figure(figsize=(30,15))

for cols in data[numeric_cols]:

    plt.subplot(221+i)

    data[cols].plot.hist()

    plt.title(cols)

    i+=1
data['Pclass'].value_counts()
# 明显有偏分布，进行取log对数变换

import math

i = 0

plt.figure(figsize=(30,15))

tmp_cols=[col for col in numeric_cols if col != 'Age']

for col in tmp_cols:

    plt.subplot(221+i)

    tmp = data[col].apply(lambda x:math.log(x+1))

    tmp.plot.hist()

    plt.title('log({})'.format(col))

    i+=1
# 离散特征分布

i = 0

plt.figure(figsize=(30,15))

for cols in data[discrete_cols]:

    plt.subplot(221+i)

    data[cols].value_counts().plot(kind='bar')

    plt.title(cols)

    i+=1
# 对id类型无法直接可视化，先暂时跳过

for col in id_cols:

    print('{}:{}'.format(col,data[col].nunique()))
# 进一步看看ticket取值与对应频数

data['Ticket'].value_counts()
# 对id类型如取值太多无法直接可视化，通常按频率从高到低排序后可视化

i = 0

plt.figure(figsize=(30,15))

for cols in data[id_cols]:

    plt.subplot(221+i)

    data[cols].value_counts().sort_values(ascending=False).plot()

    plt.title(cols)

    i+=1
# 查看特征在正负样本上的分布

grouped = data.groupby('Survived')

group0 = grouped.get_group(0)

group1 = grouped.get_group(1)
i = 0

plt.figure(figsize=(30,20))

for col in data[id_cols]:

    plt.subplot(2,2,1+i)

    group0[col].value_counts().sort_values(ascending=False).plot()

    plt.title(col)

    i+=1

    plt.subplot(2,2,1+i)

    group1[col].value_counts().sort_values(ascending=False).plot()

    plt.title(col)

    i+=1
# 数值类型特征查看分布

plt.figure(figsize=(30,20))

i = 0

for col in numeric_cols:

    plt.subplot(4,2,1+i)

    group0[col].plot.hist()

    plt.title(col)

    i+=1

    plt.subplot(4,2,i+1)

    group1[col].plot.hist()

    plt.title(col)

    i+=1

plt.figure(figsize=(30,30))

i = 0

for col in discrete_cols:

    plt.subplot(4,2,1+i)

    group0[col].value_counts().plot(kind='bar')

    plt.title(col)

    i+=1

    plt.subplot(4,2,i+1)

    group1[col].value_counts().plot(kind='bar')

    plt.title(col)

    i+=1
import seaborn as sns

sns.set(style='ticks',color_codes=True)

filled_numeric=data[numeric_cols].fillna(-1)

sns.pairplot(filled_numeric)
# 打印pearson系数

corr = filled_numeric.corr(method='pearson')

print(corr)
# 分类变量需要先encode成数值型才能求相关性

filled_discrete=data[discrete_cols].fillna('#')

for col in filled_discrete.columns:

    filled_discrete[col] = filled_discrete[col].astype('category')

cat_columns = filled_discrete.select_dtypes(['category']).columns

filled_discrete = filled_discrete.apply(lambda x: x.cat.codes)



corr=filled_discrete.corr()

print(corr)
# 整体来看特征与label之间的相关性

filled = pd.concat([filled_numeric,filled_discrete],axis=1)

corr=filled.corr(method='pearson')

print(corr)
data[string_cols].head()
# 先看文本类的特征的统计词频情况，有个初步了解

words=[{}for i in range(5)]



def count_words(s,bag):

    words_list = s.strip().split(',')

    for w in words_list:

        if w in bag:

            bag[w]+=1

        else:

            bag[w]=1



for i,col in enumerate(string_cols):

    data[col].apply(count_words,bag=words[i])

    print('###'+col+'###')

    print(pd.Series(words[i]).sort_values(ascending=False))

    print('\n')
# 可以考虑把姓名中的称谓提取做特征数据

import re

def get_title(name):

    title = re.search(' ([A-Za-z]+)\.',name)

    if title:

        return title.group(1)

    return ''



title = data['Name'].apply(get_title)

title.value_counts()
title = title.replace(['Capt','Col','Major','Dr','Rev'],'officer')

title = title.replace(['Jonkheer','Don','Sir','Countess','Dona','Lady'],'royalty')

title = title.replace(['Mrs','Mme'],'Mrs')

title = title.replace(['Miss','Mlle','Ms'],'Miss')



title.value_counts()
# cabin数据可知其首字母代表客舱类别，我们可以将其首字母提出

data['Cabin'].fillna('###',inplace=True)

data['Cabin'].value_counts()
b = data['Cabin'].apply(lambda x:'###' if len(str(x).strip())==0 else x)

b.value_counts()
cabin = b.map(lambda c:c[0])
cabin.value_counts()