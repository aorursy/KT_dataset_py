# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df.head()
df.dtypes
df.describe()
df["MSZoning"]
import matplotlib.pyplot as plt
x=df[["MSZoning","SalePrice"]]
grp=x[["MSZoning","SalePrice"]]
y=grp.groupby(["MSZoning"],as_index=False).mean()
y
y.plot(kind="bar")
import seaborn as sns
sns.boxplot(x="MSZoning",y="SalePrice",data=x)
sns.regplot(x="LotFrontage",y='SalePrice',data=df)
sns.boxplot(x="LotFrontage",y="SalePrice",data=df)
a=df[["MSSubClass","SalePrice"]]
a.head()
sns.boxplot(x="MSSubClass",y="SalePrice",data=a)
b=a.groupby(["MSSubClass"],as_index=True).mean()
b.plot(kind="bar")
sns.boxplot(x="MSSubClass",y="SalePrice",data=df)
x=df[df["LotArea"]<75000]
x=x[["LotArea","SalePrice"]]
x.head()
x["LotArea"].min()
x["LotArea"].max()
sns.regplot(x="LotArea",y="SalePrice",data=x)
x=df[["Street","SalePrice"]]
x.head()
y=x.groupby("Street",as_index=False).mean()
y
sns.boxplot(x='Street',y='SalePrice',data=x)
x=df[["Alley","SalePrice"]]
x.head()
sns.boxplot(x="Alley",y="SalePrice",data=x)
y=x.groupby("Alley",as_index=True).mean()
y
x=df[['LotShape','SalePrice']]
x.head()
sns.boxplot(x='LotShape',y='SalePrice',data=x)
y=x.groupby("LotShape",as_index=True).mean()
y
x=df[['LandContour','SalePrice']]
x.head()
sns.boxplot(x='LandContour',y='SalePrice',data=df)
