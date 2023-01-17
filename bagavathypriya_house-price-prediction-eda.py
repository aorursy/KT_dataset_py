# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
df.head()
len(df.columns)
#Features with NaN values
featurena=[fea for fea in df.columns if df[fea].isnull().sum()>1]
len(featurena)
for i in featurena:
    print(i,np.round(df[i].isnull().mean(),4),"% of missing values")
sns.heatmap(df.isnull(),xticklabels='auto',yticklabels=False)
data=df.copy()
for feature in featurena:
    data[feature]=np.where(data[feature].isnull(),1,0)
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.title(feature)
    plt.show()
numfeature=[fea for fea in df.columns if df[fea].dtype!='O']
df[numfeature].head()
yrfeature=[fea for fea in numfeature if 'Yr' in fea or 'Year' in fea]
print(yrfeature)
df.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('YrSold')
plt.ylabel('Sales Price')
for fea in yrfeature:
    plt.scatter(df[fea],df['SalePrice'])
    plt.xlabel(fea)
    plt.ylabel('Sales Price')
    plt.show()
discretefea=[fea for fea in numfeature if len(df[fea].unique())<25 and fea not in yrfeature and ['Id']]
discretefea

print(len(discretefea))
for fea in discretefea:
    df.groupby(fea)['SalePrice'].median().plot.bar()
    plt.xlabel(fea)
    plt.ylabel('Sales Price')
    plt.show()
confea=[fea for fea in numfeature if fea not in discretefea+yrfeature+['Id']]
confea
print(len(confea))
for fea in confea:
    df[fea].hist(bins=25)
    plt.xlabel(fea)
    plt.ylabel('Count')
    plt.show()
#dat=df.copy()
for fea in confea:
    dat=df.copy()
    if 0 in df[fea].unique():
        pass
    else:
        dat[fea]=np.log(dat[fea])
        dat['SalePrice']=np.log(dat['SalePrice'])
        plt.scatter(dat[fea],dat['SalePrice'])
        plt.xlabel(fea)
        plt.ylabel('Sales Price')
        plt.show()
for fea in confea:
    dat=df.copy()
    if 0 in dat[fea].unique():
        pass
    else:
        dat[fea]=np.log(dat[fea])
        dat['SalePrice']=np.log(dat['SalePrice'])
        dat.boxplot(column=fea)
        plt.ylabel(fea)
        plt.show()
catfeature=[fea for fea in df.columns if df[fea].dtype=='O']
print(len(catfeature))
for fea in catfeature:
    print('The no of categories in {} feature is {} '.format(fea,len(df[fea].unique())))
for fea in catfeature:
    dat=df.copy()
    df.groupby(fea)['SalePrice'].median().plot.bar()
    plt.xlabel(fea)
    plt.ylabel('Sales Price')
    plt.show()
