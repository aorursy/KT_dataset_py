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



        

data = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

data.head()

data.shape



tags = data.columns.values

# Any results you write to the current directory are saved as output.
data.describe()

corrmat = data.corr()

f,ax = plt.subplots(figsize=(20,9))

sns.heatmap(corrmat,vmax=.8, annot=True);
sns.set()

cols = ['SalePrice','OverallQual','GrLivArea','FullBath']

#sns.pairplot(data[cols],size = 2.5)

#df = pd.DataFrame(data['OverallQual','SalePrice'],columns=["Overallqual",'SalePrice'])

print(data[cols])

sns.swarmplot(x=data["OverallQual"],y=data["SalePrice"])

plt.show();
sns.boxplot(x=data["OverallQual"],y=data["SalePrice"])
sns.violinplot(x=data["FullBath"],y=data["SalePrice"],inner=None)
ax=sns.stripplot(x=data["FullBath"],y=data["SalePrice"])

ax=sns.violinplot(x=data["FullBath"],y=data["SalePrice"],jitter=True,inner=None)
