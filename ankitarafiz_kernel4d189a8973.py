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
df=pd.read_csv('/kaggle/input/depression/b_depressed.csv')
df.columns
df['incoming_salary'].corr(df['depressed'])
df.head()
df.count()
df.describe()
df.isnull().sum()
import matplotlib.pyplot as plt
plt.hist(df['depressed'])
df2=df[df['depressed']==1]
plt.hist(df2['depressed'])
print(df['incoming_salary'].mean())
print(df2['incoming_salary'].mean())
print(df['Age'].mean())
print(df2['Age'].mean())
print(df['education_level'].mean())
print(df2['education_level'].mean())
print(df['Number_children'].mean())
print(df2['Number_children'].mean())
print(df['lasting_investment'].mean())
print(df2['lasting_investment'].mean())
plt.hist(df2['sex'])
df['total_members'].corr(df['depressed'])
print(df['total_members'].mean())
print(df2['total_members'].mean())
plt.scatter(df['gained_asset'],df['durable_asset'])
plt.scatter(df['depressed'],df['durable_asset'])
df['durable_asset'].corr(df['depressed'])

df['durable_asset'].corr(df['gained_asset'])

