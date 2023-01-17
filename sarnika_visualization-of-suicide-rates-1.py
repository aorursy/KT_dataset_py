# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use('seaborn-whitegrid')

sns.set_style('whitegrid')

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv')

df.head()
df.info()
df.describe()
print('missing rows in each column: \n')

c=df.isnull().sum()

print(c[c>0])
df.drop(columns=['HDI for year'])
plt.figure(figsize=(20,10))

df['country'].value_counts().plot.bar()
plt.figure(figsize=(10,8))

df['year'].value_counts().sort_index().plot.bar()
plt.figure(figsize=(6,4))

df['sex'].value_counts().sort_index().plot.bar()