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
data=pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv')
df=data.copy()
df.head(20)
df.tail()
df.shape
df.info()
df.columns
df['age']
kat_df=df.select_dtypes(include=['object'])
kat_df
data_1=df[['year','country-year','country']]
data_1.sample(45)
df.drop('country-year',inplace=True,axis=1)
df.columns
df.rename(columns={' gdp_for_year ($) ':'gdp_for_year ($)'})
null_HDI=df['HDI for year'].isnull().sum()
null_HDI
null_HDI*100/len(df)
df.drop('HDI for year',inplace=True,axis=1)
df.describe().T
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("ggplot")
f,ax=plt.subplots(figsize = (7,7))
# corr() is actually pearson correlation
sns.heatmap(df.corr(),annot= True,linewidths=0.5,fmt = ".1f",ax=ax)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title('Correlation Map')
plt.show()
