# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from sklearn.cluster import KMeans



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

#read the csv file

data_train=pd.read_csv("../input/train.csv")

data_train.info()

#read the csv file

data_train.head()
#data_train['SaleType'].value_counts()

sns.boxplot(x='SaleType',y='SalePrice',hue='SaleCondition',data=data_train,width=0.5)





#g = sns.FacetGrid(data_train, col="SaleCondition", size=4, aspect=.7)

#(g.map(sns.boxplot, "SaleType", "SalePrice").despine(left=True).add_legend(title="smoker"))  
data_train['SalePrice'].hist(bins=20)
sns.boxplot(x='MSZoning',y='SalePrice',data=data_train,width=0.3)

data_train['MSZoning'].value_counts()
ax=sns.boxplot(y='Neighborhood',x='SalePrice',data=data_train,width=0.3)