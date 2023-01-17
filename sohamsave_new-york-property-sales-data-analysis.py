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
df= pd.read_csv('../input/nyc-rolling-sales.csv')
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df.head()
df.info()
df.drop(['EASE-MENT','APARTMENT NUMBER','Unnamed: 0','ZIP CODE'],axis=1, inplace=True)
def sale(string):
    if ('-' in string):
        return np.NaN
    else:
        return int(string)
def year_built(string):
    if string==0:
        return np.NaN
    elif ('-' in string):
        return np.NaN
    else:
        return int(string)
df['SALE PRICE']=df['SALE PRICE'].map(lambda x: sale(x))
df['YEAR BUILT']=df['YEAR BUILT'].map(lambda x: int(x))
df['YEAR SOLD']=df['SALE DATE'].map(lambda x: x[0:4])
df['YEAR SOLD']=df['YEAR SOLD'].map(lambda x: int(x))
df['MONTH SOLD']=df['SALE DATE'].map(lambda x: x[5:7])
df['MONTH SOLD']=df['MONTH SOLD'].map(lambda x: int(x))
df['GROSS SQUARE FEET']=df['GROSS SQUARE FEET'].map(lambda x: year_built(x))
df['LAND SQUARE FEET']=df['LAND SQUARE FEET'].map(lambda x: year_built(x))
df['YEAR BUILT'][df['YEAR BUILT']==0]=np.NaN
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df.dropna(axis=0,inplace=True)
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df.info()



sns.countplot('BOROUGH',data=df,palette='Set2')
plt.title('Sales per Borough')
sns.barplot(y='RESIDENTIAL UNITS', x='BOROUGH',data=df, palette='coolwarm', ci=None)
plt.title('Sales per borough_Residential')
sns.barplot(y='COMMERCIAL UNITS', x='BOROUGH',data=df, palette='coolwarm', ci=None)
plt.title('Sales per borough_Commercial')
sns.countplot(x='YEAR SOLD', data=df, palette='rainbow')
plt.title('Sales Rate from 2016-2017')
sns.barplot(x='YEAR SOLD', y='SALE PRICE', hue='BOROUGH', data=df, palette='rainbow', ci=None)
plt.title('Sales per Borough from 2016-2017')
plt.figure(figsize=(20,5))
sns.barplot(x='MONTH SOLD', y='SALE PRICE', hue='BOROUGH', data=df, palette='rainbow', ci=None)
plt.title('Sales per Borough from 2016-2017')
plt.legend(loc='lower center',bbox_to_anchor=(0.5,-0.5))
plt.figure(figsize=(20,5))
sns.countplot('MONTH SOLD', hue='YEAR SOLD', data=df, palette='RdBu_r')
