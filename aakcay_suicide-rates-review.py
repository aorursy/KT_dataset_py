# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')
data.info()
data.corr
f,ax=plt.subplots(figsize=(18,18))

sns.heatmap(data.corr(),annot=True,linewidths=.5,fmt='.1f',ax=ax)

plt.show()


data.dtypes
data.head(20)
data.population .plot(kind="hist",bins=50,figsize=(12,12))
data.year.plot(kind='hist',bins=50,figsize=(12,12))
