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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv('../input/train.csv')

df.head()
df.shape
# Here we identify the Target variable and Predictor variable

Target_var=df['SalePrice']

Pred_var=df.drop(['SalePrice'],axis=1)
Target_var.describe()
#histogram

sns.distplot(Target_var)
#skewness and kurtosis

print("Skewness: %f" % Target_var.skew())

print("Kurtosis: %f" % Target_var.kurt())
def miss_data(df):

    Total=df.isna().sum().sort_values(ascending=False)

    Percentage=((df.isna().sum()/df.shape[0])*100).sort_values(ascending=False)

    return pd.concat([Total,Percentage],keys=['Total','Percentage'],axis=1)

a=miss_data(df)

a
most_missed_data =df[['PoolQC','MiscFeature','Alley','Fence','FireplaceQu']]

most_missed_data.dtypes
def drop(df):

    for index,row in a.iterrows():

        if row[1]>30:

            df.drop(index,axis=1,inplace=True)

b=drop(df)

b
df.shape
miss_data(df)