# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style('whitegrid')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/to Lior.csv', nrows=252)
df.head()
df.columns = ['binding_energy', 'cps', 'f1', 'f2', 'f3', 'f4', 'f5', 'f_sum']
ax = df.plot(x='binding_energy', y='cps', kind='scatter', c='black', figsize=(10, 6))
df.plot(x='binding_energy', y=['f1', 'f2', 'f3', 'f4', 'f5',], kind='line', c='red', ax=ax)
df.plot(x='binding_energy', y='f_sum', kind='line', c='blue', ax=ax)
