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
# Load the librarys
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
#Load the training and test data sets
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
# To know about data set(No.of. observations and features)
print('Training Data:' , train_data.shape )
print('Test Data:' , train_data.shape )
#Looking the Type of Data
print(train_data.info())

#Looking unique values
print(train_data.nunique())
#Looking the sample data
print(train_data.head())
from matplotlib.colors import ListedColormap
corr_plot = train_data.corr()
plt.figure(figsize=(14,12))
d=sns.heatmap(corr_plot.astype(float).corr(),linewidths=0.1,vmax=1.0,cmap=ListedColormap(['blue', 'yellow', 'green']),
            square=True,  linecolor='white')

plt.show()
fig, ax = plt.subplots(3,1, figsize=(12,14))
sns.countplot(x="Neighborhood",data=train_data, orient=45, ax=ax[0])
sns.pointplot(x="Neighborhood", y ="MSSubClass",data=train_data, ax=ax[1])
sns.violinplot(x="Neighborhood", y="SalePrice", data=train_data,ax=ax[2])
leg = ax[0].rotation=45
leg = ax[1].rotation=45
plt.xticks(rotation=45)
plt.show
