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
df = pd.read_csv('../input/Weekly Sales all product.csv')
df.head()
df.info()
df.describe()
df.isnull().sum()
var = df.groupby('Year').Actual.sum()
var.plot(kind='bar')
var1 = df.groupby('Month').Actual.sum()
var1.plot(kind='bar')
df['AverageLast10Weeks'] = df[['W1','W2','W3','W4','W5','W6','W7','W8','W9','W10']].mean(axis=1)
var2 = df.groupby('SKU',sort=True).Actual.sum().reset_index()
var2 = var2.sort_values('Actual', ascending=False)
var2.set_index('SKU',inplace=True)
var2[0:20].plot(kind='bar')