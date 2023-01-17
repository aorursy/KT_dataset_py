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
df = pd.read_csv('../input/bank.csv')
df.head(10)
df.shape
df.info
df.describe()
df['education'].value_counts()
import matplotlib.pyplot as plt
df.select_dtypes(include=['int64']).columns
numeric_cols =  [col for col in df.columns if df[col].dtype != object]
numeric_cols
plt.hist(df['age'],bins = 20)
df[numeric_cols].hist(bins=20)
plt.show()
plt.boxplot(df['balance'])
plt.show()
plt.hist(df[(df['balance'] < 10000) &(df['balance'] > 0) ]['balance'])
plt.show()

plt.hist(df['campaign'])
plt.show()
plt.boxplot(df['campaign'])
plt.show()
plt.hist(df[df['campaign'] < 9]['campaign'])
plt.show()
import seaborn as sns
df['education'].value_counts()
plt.scatter(df['campaign'],df['age'])
plt.show()
sns.pairplot(data = df,vars = numeric_cols[:3])
plt.show()
sns.pairplot(data = df,vars = numeric_cols[:3],hue = 'deposit')
plt.show()
#Explaining Explanatory Power
df['deposit_1'] = df['deposit'].map({"yes":1,"no":0})
sns.boxplot(x='deposit_1',y='duration',data=df)
plt.show()
