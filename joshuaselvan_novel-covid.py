# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sn

import matplotlib

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
corona=pd.read_csv('/kaggle/input/eda-for-novel-covid19/world-corona-reports - Copy.csv')

corona
coronas=corona.head(10)

coronas
corona.shape
corona.columns
corona['Total\rCases'].sum()
corona.nunique(axis=0)
corona.nunique(axis=1)
corona.isnull().sum()
first_column = corona.iloc[:, 1]

first_column
second_column = corona.iloc[:,0]

second_column
df=pd.concat([second_column,first_column], axis=1)

df
heal=df.head(10)

heal
heal.plot(x ='Country', y='Total\rCases', kind = 'scatter')
heal.plot(x ='Country', y='Total\rCases', kind = 'bar')
nen=df.set_index('Country')

nen
nen.max()
nen.min()
box=corona.head(10)

box
boxplot = box.boxplot(column=['New\rCases'])
boxplot = box.boxplot(column=['New\rDeaths'])
boxplot = box.boxplot(column=['Active\rCases'])
corona.describe()
corona.corr()
corrMatrix = corona.corr()

sn.heatmap(corrMatrix, annot=True)

plt.show()