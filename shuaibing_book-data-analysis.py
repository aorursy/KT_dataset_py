# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style("darkgrid")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/bookserror-correction/books(error_correction).csv",encoding="utf8")

data.head()
plt.figure(figsize=(16,6))

sns.distplot(data.average_rating,rug_kws={"color":"green"},rug=True)
# 我们看一下所有书籍语言分布情况

plt.figure(figsize=(16,6))

sns.countplot(data.language_code)
plt.figure(figsize=(16,6))

sns.boxplot(data["# num_pages"])
# 我们可以看下评分数量与评论数量是否有关联

plt.figure(figsize=(16,6))

sns.pointplot(x="ratings_count",y="text_reviews_count",data=data)

plt.xticks(range(0,7000,700),(600000*i for i in range(10)))
# 多变量分析,热力图显示变量之间的关系

corrArray = data.corr()

plt.figure(figsize=(6,6))

sns.heatmap(corrArray)
# 我们可以看下评分与书本页数是否有关联

plt.figure(figsize=(16,6))

sns.pointplot(x="average_rating",y="# num_pages",data=data)
plt.figure(figsize=(16,16))

sns.pairplot(data)