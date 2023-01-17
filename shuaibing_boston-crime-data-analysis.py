# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# -*- coding: utf-8 -*-

data = pd.read_csv("/kaggle/input/crime.csv",encoding = "gb18030")
data.head()
data.shape
data.SHOOTING.unique()
data.SHOOTING = data.SHOOTING.fillna("N")

sns.countplot(data.SHOOTING,)
plt.figure(figsize=(16,6))

sns.countplot(data.OFFENSE_CODE_GROUP)

plt.xticks(rotation=-90)
plt.figure(figsize=(16,6))

sns.countplot(data.DISTRICT)

plt.xticks(rotation=-90)
# 下面进行统计不同年月发生犯罪的频率差异

fig = plt.figure(figsize=(16,12))

sns.countplot(data.YEAR,ax=fig.add_subplot(221))

sns.countplot(data.MONTH,ax=fig.add_subplot(222))

sns.countplot("MONTH",data=data,hue="YEAR",ax=fig.add_subplot(212))
plt.figure(figsize=(16,8))

sns.countplot(data.HOUR)
# 我们对地点绘制成散点图,部分坐标数据可能有误，故设置x、y轴坐标范围

plt.figure(figsize=(12,12))

sns.scatterplot(x="Lat",y="Long",data=data)

plt.axis([42.2,42.45,-71.2,-70.95])