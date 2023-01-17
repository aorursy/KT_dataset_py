# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_style("whitegrid")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/properties.csv')
df.head()
df.info()
df.describe()
df.isnull().sum()
df = df.drop(['Unnamed: 0'], axis=1)
df = df[(df['lat'] != -999 ) & (df['lng'] != -999)]
sns.pairplot(df)
sns.distplot(df['Price ($)'])
sns.heatmap(df.corr())
