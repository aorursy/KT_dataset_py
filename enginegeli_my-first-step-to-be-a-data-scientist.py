# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_set = pd.read_csv ('../input/world-happiness/2015.csv')

data_set.info()

data_set.describe()

data_set.columns

data_set.head(10)
#Corelation is relationship between features



data_set.corr()



#Corelation map



f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data_set.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()



data_set.columns = [ each.lower() for each in data_set.columns] 



data_set.columns = [each.split()[0]+"_"+each.split()[1] if(len(each.split())>1) else each for each in data_set.columns]

data_set.rename(columns={'economy_(gdp':'economy'}, inplace=True)

data_set.rename(columns={'trust_(government':'trust'}, inplace=True)

data_set.columns

# Scatter Plot 

# x = attack, y = defense

data_set.plot(kind='scatter', x='freedom', y='happiness_score',alpha = 0.5,color = 'red')

plt.xlabel('Freedom')              # label = name of label

plt.ylabel('Happiness Score')

plt.title('Freedom & Happiness Score Scatter Plot')     
# Histogram

# bins = number of bar in figure

data_set.economy.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.xlabel('Economy')

plt.show()
data_set[np.logical_and(data_set['trust']>0.4, data_set['economy']>1.3 )]
