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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
test.info()
train.drop('MiscFeature', axis = 1, inplace = True)

test.drop('MiscFeature', axis = 1, inplace = True)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.boxplot(x = "LotConfig", y= 'LotFrontage', data = train)
sns.boxplot(y="Fireplaces", x="FireplaceQu", data = test)
train["FireplaceQu"].count()
train["Fence"].value_counts()
train["Alley"].value_counts()
sns.jointplot(x = "LotFrontage", y = "SalePrice", data = train)
#go through and clean all data, remove outliers and NaNs. Go through columns and see what correlates well with Sale Price