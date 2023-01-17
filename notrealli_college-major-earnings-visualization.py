!pip install pandas --upgrade
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
%matplotlib inline

from pandas.plotting import scatter_matrix 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
grads = pd.read_csv('/kaggle/input/college-earnings-by-major/recent-grads.csv')
grads.head(10)
grads.tail()
grads.describe()
grads.describe(include='object')
grads.info()
grads.dropna(inplace=True)
grads.info()
grads.plot(kind='scatter', x='Total', y='Median', xlabel='Major population', ylabel='Median income', title='Total vs. Median', xlim=(0, 100000))
grads.plot(kind='scatter', x='Total', y='Unemployment_rate', title='Total vs. Unemployment rate', xlim=(0, 100000))
grads.plot(kind='scatter', x='Full_time', y='Median', title='Full-time employment vs. Median income', xlim=(0, 50000), ylim=(20000, 80000))
grads.plot(kind='scatter', x='ShareWomen', y='Unemployment_rate', title='Share of women in Major vs. Unemployment rate', ylim=(0, 0.125))
grads.plot(kind='scatter', x='Men', y='Median', title='Men in Major vs. Median income')
grads.plot(kind='scatter', x='Women', y='Median', title='Women in Major vs. Median income')
grads['Total'].describe()
grads['Total'].hist(bins=10, range=(0, 400000))
grads['Median'].describe()
grads['Median'].hist(bins=11, range=(15000, 110000))
employment_rate = (grads['Employed'] / grads['Total'])
employment_rate.describe()
employment_rate.hist(bins=10, range=(0, 1))
grads['ShareWomen'].describe()
grads['ShareWomen'].hist(bins=10, range=(0, 1))
scatter_matrix(grads[['Sample_size', 'Median', 'Unemployment_rate']], figsize=(15,15))
grads[:20].plot.bar(x='Major', y='ShareWomen', figsize=(15, 8))
grads[:20].plot.bar(x='Major', y='Unemployment_rate', figsize=(15, 8))