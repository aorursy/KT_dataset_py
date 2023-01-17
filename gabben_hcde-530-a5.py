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
import pandas as pd
df = pd.read_csv('../input/poetry-foundation-poems/PoetryFoundationData.csv')
df.head()
df.tail()
df.iloc[:50]
df.iloc[-50:]
df.drop(columns=['Unnamed: 0'])
df.info()
df['Tags'].unique()
len(df['Tags'].unique().tolist())
df.Tags.drop_duplicates()
tags = df['Tags'].tolist() #make a list with your column values
tags
tags_list = pd.Series(tags)
tags_list.str.split(',')
df.Tags.mode()
df['Tags'].value_counts().idxmax()
df.Poet.mode()
n = 10
df['Tags'].value_counts()[:n].index.tolist()
df['Poet'].value_counts()[:n].index.tolist()
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
df['Poet'].value_counts().plot(ax=ax, kind='bar')

n = 20
fig, ax = plt.subplots()
df['Poet'].value_counts()[:n].plot(ax=ax, kind='bar')