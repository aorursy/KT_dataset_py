# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/vgsales.csv')
data.columns
data.info()
data.describe()
data.head()
data.Year.plot(kind='hist', bins=50)
plt.show()
count = 0
for index,value in data [['Publisher']][0:].iterrows():
    if value[0] == 'Nintendo':
        count = count + 1
        print(index, "=" ,value[0])
print("Number of Nintendo in toplist", count)
data.corr()
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data[np.logical_and(data['Year']>1999, data['Global_Sales']>25 )]