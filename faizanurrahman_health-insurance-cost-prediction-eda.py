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
df = pd.read_csv('../input/insurance.csv')
df.head()
df.info()
df.describe()
df.isnull().sum()
sns.countplot(x='smoker',data=df,palette='viridis')
sns.countplot(x='children',data=df,palette='viridis')
df.age.nunique()
sns.countplot(x='region',data=df,hue='sex',palette='viridis')
by_region = df.groupby('region').charges.sum()
by_region.plot(kind='bar')
by_sex = df.groupby('sex').charges.sum()
by_sex.plot(kind='bar')
by_nofchildren = df.groupby('children').charges.sum()
by_nofchildren.plot(kind='bar')
by_smoker = df.groupby('smoker').charges.sum()
by_smoker.plot(kind='bar')
