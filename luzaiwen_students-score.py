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
import matplotlib.pyplot as plt
df = pd.read_csv("../input/StudentsPerformance.csv")
df.head()
df.isnull().sum()
df.describe()
df.info()
passmark=60
df['Math_PassStatus'] = np.where(df['math score']>=passmark, 'pass', 'not')
df['Reading_PassStatus'] = np.where(df['reading score']>=passmark, 'pass', 'not')
df['Writing_PassStatus'] = np.where(df['writing score']>=passmark, 'pass', 'not')
df2 = pd.get_dummies(df,drop_first=False)
import seaborn as sns
df3 = df2[df2.columns[~df2.columns.isin(['math score','reading score','writing score'])]]
cor = df3.corr()[['Math_PassStatus_pass','Reading_PassStatus_pass','Writing_PassStatus_pass']][:-6]
cor
plt.figure(figsize=(10,10))
cmap = sns.cubehelix_palette(start = 1, rot = 3, gamma=0.8, as_cmap = True)
sns.heatmap(cor,cmap='rainbow')





