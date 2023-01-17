# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import tensorflow as tf # tensorflow

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# labels of properties

properties_csv = pd.read_csv("../input/properties_2016.csv", low_memory=False)

print(properties_csv.shape)
print(properties_csv.sample(5))
print(train_csv.sample(5))
train_csv = pd.read_csv("../input/train_2016.csv", low_memory=False)

print(train_csv.shape)
plt.figure(figsize=(10,8))

sns.distplot(train_csv.logerror.values, bins=50, kde=False)

plt.xlabel('logerror', fontsize=12)

plt.show()
plt.figure(figsize=(8,8))

plt.scatter(range(train_csv.shape[0]), np.sort(train_csv.logerror.values))

plt.xlabel('index', fontsize=12)

plt.ylabel('logerror', fontsize=12)

plt.show()
(train_csv['parcelid'].value_counts().reset_index())['parcelid'].value_counts()
#这些代码还是有问题的,需要debug

id_counts = train_csv['parcelid'].value_counts()

id_counts.where(id_counts>1).dropna(how='any')

for row in id_counts:

    print(row)
merge_csv = train_csv.merge(properties_csv)

merge_csv.shape
merge_csv['parcelid'].value_counts()