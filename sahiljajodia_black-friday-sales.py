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
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
df = pd.read_csv('../input/BlackFriday.csv')
df['Product_Category_2'].fillna(0.0, inplace=True)
df['Product_Category_3'].fillna(0.0, inplace=True)
df.head()
df.Marital_Status.value_counts()
df.info()
df1 = df[:50000]
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap = 'viridis')
plt.figure(figsize=(8, 6))
sns.boxenplot(x='Age', y='Purchase', data=df)
sns.catplot(x='City_Category', y='Purchase', data=df, kind='violin')
sns.catplot(x='City_Category', y='Purchase', data=df, kind='violin', hue='Gender', split=True)
sns.catplot(x='Gender', y='Purchase', data=df, kind='violin')
plt.figure(figsize=(10, 6))
sns.boxplot(x='Occupation', y='Purchase', data=df)
sns.countplot(x='Gender', data=df)

sns.countplot(x='City_Category', data=df)
plt.figure(figsize=(10, 6))
sns.barplot(x='Occupation', y='Purchase', data=df, estimator=np.median)
plt.figure(figsize=(10, 6))
sns.barplot(x='Occupation', y='Purchase', data=df, palette="Blues_d")
sns.countplot(x='City_Category', hue='Gender', data=df)
plt.figure(figsize=(15, 8))
sns.countplot(x='Occupation', hue='Marital_Status', data=df)

