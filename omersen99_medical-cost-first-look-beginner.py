# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/insurance.csv')
data.columns
data.describe()
data.info

data.head(5)
data.corr()
korelasyn = plt.subplots(figsize=(8,8))
sns.heatmap(data.corr(), annot=True, linewidths=.3)
plt.show()
data.children.plot(kind='hist', bins=10, figsize=(8,8))
plt.show()
data.plot(kind='scatter', x = 'age', y='charges', alpha=1, color='black')
plt.xlabel('age')
plt.ylabel('charges')
plt.title('yaşa göre harcamalar')
data["children"].describe()
ortalamakaçcocuk=np.mean(data.children)
print("ortalamakaçcocuk=",ortalamakaçcocuk)
x = data['children']>ortalamakaçcocuk
data[x]
datawithoutgender=data.drop('sex', axis=1)
datawithoutgender
data.groupby(['age'])[['charges']].agg(['mean','median','count'])
