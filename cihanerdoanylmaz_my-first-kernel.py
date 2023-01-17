# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns #visualisation tool

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

df=pd.read_csv('../input/world-bank-trade-costs-and-trade-in-services/trade-in-services-csv.csv')

df.info()
df.head()
#correlation map

df.corr()

f, ax = plt.subplots(figsize = (13,13))

sns.heatmap(df.corr(),annot =True, linewidth = 5, fmt = ' .1f', ax = ax)
df.BOP.plot(kind = 'line', color = 'r', label = 'Balance of Payments', linewidth = 2, alpha = 0.5, grid = True, linestyle = ':')

df.VALUE.plot(color = 'g', label = 'Value', linewidth = 2, alpha = 0.5, grid = True, linestyle = ':')

plt.legend(loc = 'upper right')

plt.xlabel('x')

plt.ylabel('y')

plt.title('Line Plot')
#Scatter Plot

df.plot(kind = 'scatter', x = 'BOP', y = 'VALUE', alpha = 0.5, color = 'r')

#plt.scatter alternative method
#histogram

df.YEAR.plot(kind = 'hist', bins = 50, figsize =(10,10))

plt.show()
#bop_z = df['BOP'] > 982

#df[np.logical_and(df['BOP'] > 982, df['VALUE'] > 200000)]

df[(df['BOP'] > 982) & (df['VALUE'] > 200000)]