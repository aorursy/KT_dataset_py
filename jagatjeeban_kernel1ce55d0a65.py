# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv')
data.head()
data.info()
data.describe()
data['country'].unique()
import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style('whitegrid')
sns.boxplot(data = data,x = 'age',y = 'suicides_no')
sns.pairplot(data,hue = 'age',palette='rainbow')
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.distplot(data['suicides_no'].dropna(),kde=False,bins=30)
plt.figure(figsize=(12, 7))

sns.boxplot(x='age',y='suicides_no',data=data,palette='winter')