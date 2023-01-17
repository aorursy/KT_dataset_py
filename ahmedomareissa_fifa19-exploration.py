# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

import seaborn as sns

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df  = pd.read_csv('/kaggle/input/fifa19/data.csv')

pd.options.display.max_columns = 999

df.head()
df.info()
comp = (df.count() / df.shape[0]).sort_values()

plt.figure(figsize = (5,20))

plt.grid()

plt.barh(width  = comp.values , y = comp.index)

plt.ylabel('column')

plt.xlabel('% of non-null values');
plt.figure(figsize = (40,10))

df.boxplot(by = 'Age' , column = 'Overall') 

plt.title('')

plt.ylabel('Rating');
df = df[df['Value'].str[1:-1]!=''] 

df = df[~df['Value'].isnull()] 

mult = [1000000 if i == 'M' else 1000 for i in df.Value.str[-1]]

df['value'] = df.Value.str[1:-1].astype(float) * mult

df.plot.scatter(y = 'value' , x = 'Overall') 

plt.yscale('log')

plt.xlabel('Rating')

plt.ylabel('Value (Log)');