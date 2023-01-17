# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/windows-store/msft.csv')
print(df.info())
print(df.describe())
df
df.dropna()
plt.rcParams['figure.figsize'] = (20, 15)
df.Rating.hist(by=df.Category)
x = df['Category']
plt.rcParams['figure.figsize'] = (30, 5)
sns.countplot(x)
plt.title('Count of Category')
plt.xlabel('Category')
plt.ylabel('Count')
sns.despine()
plt.rcParams['figure.figsize'] = (20, 8)
x = df['Rating']
plt.title('Distribution of Rating')
sns.countplot(x)
plt.rcParams['figure.figsize'] = (10,4)
sns.heatmap(df.corr(), annot=True)