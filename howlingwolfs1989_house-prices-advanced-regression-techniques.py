# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

%matplotlib inline
sns.set(style='ticks', palette='Paired', font_scale=1)
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df = train.copy()
df.head()
df.drop(['Id'],inplace=True, axis=1)
plt.figure(figsize=(10, 5))

sns.distplot(df['SalePrice']);
plt.figure(figsize=(10, 5))

sns.distplot(df['LotArea']);
plt.figure(figsize=(10, 5))

sns.kdeplot(df['SalePrice'], shade=True, color="r");

sns.kdeplot(df['LotArea'], shade=True, color="b");
def plot_sales_price(col, rot=0):

    

    plt.figure(figsize=(15, 10))

    plt.subplot(221)

    plt.xticks(rotation=rot)

    sns.barplot(x=col, y='SalePrice', data=df, ci=None);

    

    plt.subplot(222)

    plt.xticks(rotation=rot)

    sns.boxplot(x=col, y='SalePrice', data=df);
plot_sales_price('MSSubClass')
plot_sales_price('MSZoning')
plot_sales_price('Street')
plot_sales_price('Alley')
plot_sales_price('LotShape')
plot_sales_price('LandContour')
plot_sales_price('Utilities')
plot_sales_price('LotConfig')
plot_sales_price('LandSlope')
plot_sales_price('Neighborhood', rot=45)
plot_sales_price('Condition1')
plot_sales_price('Condition2')
plot_sales_price('BldgType')
plot_sales_price('HouseStyle')
plot_sales_price('OverallQual')
plot_sales_price('OverallCond')
plot_sales_price('RoofStyle')
plot_sales_price('Exterior1st', rot=45)
plot_sales_price('Exterior2nd', rot=45)
plot_sales_price('MasVnrType')
plot_sales_price('ExterQual')
plot_sales_price('ExterCond')
plot_sales_price('Foundation')
plot_sales_price('BsmtQual')
plot_sales_price('BsmtCond')
plot_sales_price('BsmtExposure')
plot_sales_price('BsmtFinType1')
plot_sales_price('BsmtFinType2')
plot_sales_price('Heating')
plot_sales_price('HeatingQC')
plot_sales_price('CentralAir')
plot_sales_price('Electrical')
plot_sales_price('KitchenQual')
plot_sales_price('Functional')
plot_sales_price('FireplaceQu')
plot_sales_price('GarageType')
plot_sales_price('GarageFinish')
plot_sales_price('GarageQual')
plot_sales_price('GarageCond')
plot_sales_price('PavedDrive')
plot_sales_price('PavedDrive')
plot_sales_price('PoolQC')
plot_sales_price('Fence')
plot_sales_price('MiscFeature')
plot_sales_price('SaleType')
plot_sales_price('SaleCondition')
plt.figure(figsize=(50, 50))

cor = df.corr()

sns.heatmap(cor, annot=True);