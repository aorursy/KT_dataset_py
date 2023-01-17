# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt# data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sn
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/haberman.csv')
df.head()
df.columns
df['age']=df['30']
df['year_op']=df['64']
df['axil_node']=df['1']
df['survival']=df['1.1']
df.drop('1.1',axis=1,inplace=True)
df.head()
sn.countplot(x='survival',data=df)
sn.boxplot(x='survival',y='age',data=df)
plt.figure(figsize=(15,6))
sn.countplot(x='age',data = df)
sn.distplot(a=df['age'],kde=True)
df.columns
plt.figure(figsize=(10,6))
sn.violinplot(x='survival',y='axil_node',data=df)
sn.stripplot(y='axil_node',x='survival',data=df,jitter=True)
