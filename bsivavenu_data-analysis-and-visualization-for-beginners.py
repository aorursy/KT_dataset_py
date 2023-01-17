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
train_df = pd.read_csv("../input/train.csv", parse_dates=["activation_date"])
test_df = pd.read_csv("../input/test.csv", parse_dates=["activation_date"])
train_df.head()
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(train_df.deal_probability.values)
plt.figure(figsize=(8,6))
plt.scatter( range(train_df.shape[0]),np.sort(train_df['deal_probability'].values))
plt.show()
plt.figure(figsize=(12,8))
sns.barplot(y=train_df.region, x="deal_probability", data=train_df)
train_df.city.value_counts()
plt.figure(figsize=(12,8))
sns.barplot(x=train_df.parent_category_name, y="deal_probability", data=train_df)
plt.figure(figsize=(12,8))
sns.boxplot(x="parent_category_name", y="deal_probability", data=train_df)
plt.ylabel('Deal probability', fontsize=12)
plt.xlabel('Parent Category', fontsize=12)
plt.title("Deal probability by parent category", fontsize=14)
plt.xticks(rotation='vertical')
plt.show()
plt.figure(figsize=(15,5))
sns.countplot(x="category_name",data=train_df)
plt.xticks(rotation='vertical')
plt.show()
plt.figure(figsize=(15,5))
sns.countplot(x="user_type",data=train_df)
plt.show()