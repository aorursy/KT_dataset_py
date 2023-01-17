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
!pip install vaex==2.5.0 

!pip install tensorflow==2  #For enabling vaex to operates 
import vaex

df = vaex.read_csv('/kaggle/input/ngc-628-7793-krumholz-2015/opencluster.tsv', delimiter = ';')
df.head()
df.columns
df.col.AV_84
df.AV_84
df['AV_84']
df.data.AV_84
df.evaluate(df.AV_84)
df.describe()
df['Field'].unique()
select = df[df['Field'] == 'NGC_7793e_l']
select.head(5)
import matplotlib.pyplot as plt

import seaborn as sns
index = select.evaluate(select['AV_84'])
plt.hist(index, bins = 15)

plt.title('AV_84')

plt.xlabel('AV_84')

plt.ylabel('n')
sns.distplot(index, bins = 15)

plt.title('AV_84')

plt.xlabel('AV_84')

plt.ylabel('n')
select.plot1d(select['AV_84'])
x = select.evaluate(select['logT_84'])

y = select.evaluate(select['AV_84'])
plt.scatter(x,y, s = 2, color = 'r')

plt.title('logT vs AV')

plt.xlabel('logT')

plt.ylabel('AV')