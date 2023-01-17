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
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import numpy as np

%matplotlib inline
df = pd.read_csv('../input/agricuture-crops-production-in-india/produce.csv')
df1 = pd.read_csv('../input/agricuture-crops-production-in-india/datafile (1).csv')
df2 = pd.read_csv('../input/agricuture-crops-production-in-india/datafile (2).csv')
df3 = pd.read_csv('../input/agricuture-crops-production-in-india/datafile (3).csv')
df4 = pd.read_csv('../input/agricuture-crops-production-in-india/datafile.csv')
df.head()
df.describe(include='all')
#(df) produce   -- Agriculture Production in india 2005-2014

#(df1) datafile1 -- Various Crops cultivation and Production

#(df2) datafile2 -- Crops Production from 2006-2011

#(df3) datafile3 -- Recommended Zone Of Crops Cultivation And Production

#(df4) datafile  -- Agriculture production in india 2004-2012

df.isnull().any()
print(df.isnull().sum())
print(df1.isnull().sum())
df1.dtypes
print(df2.isnull().sum())
print(df3.isnull().sum())
df3.dtypes
print(df4.isnull().sum())
temp1 = pd.crosstab(df1['State'], df1['Crop'])

temp1.plot(kind='bar', stacked=True, figsize = (16,10))
corr = df2.corr()

f, ax = plt.subplots(figsize=(15, 8))

sns.heatmap(corr, annot=True, cmap="RdPu")

plt.show()
ax=df1.set_index('State','Crop').plot(kind='bar',color=['red','c'],figsize=(10,2));

ax.axvline(1000,color = '.7',linestyle= "--",linewidth=1)

ax.set_xlabel('State')

ax.set_ylabel('');
sns.jointplot(x='Yield (Quintal/ Hectare) ', y='Cost of Cultivation (`/Hectare) C2', data=df1[df1['Yield (Quintal/ Hectare) '] > 10], kind="reg", space=0, color="g")