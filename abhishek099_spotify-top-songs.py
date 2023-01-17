# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/top50spotify2019/top50.csv', encoding = 'latin-1')
data.head(10)
data= data.drop('Unnamed: 0', axis = 1)
data.isnull().sum()
data
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize = (10,10))
sns.countplot(data['Genre'])
plt.xticks(rotation = 90)
data['Genre'].value_counts()
data['Artist.Name'].value_counts()
import warnings
warnings.filterwarnings('ignore')
f, ax = plt.subplots(figsize = (12,12))
sns.heatmap(data.corr(), annot = True, linewidth = .5, fmt = '.1f', ax = ax)
sns.boxplot(y = data['Popularity'])

plt.figure(figsize = (15,15))
sns.jointplot(x = data.loc[:,'Beats.Per.Minute'], y = data.loc[:,'Popularity'], kind = 'regg', color = '#ce1414')
plt.show()
sns.jointplot(data.loc[:, 'Loudness..dB..'], data.loc[:, 'Popularity'], kind = 'regg', size = 10)
plt.show()
plt.figure(figsize = (12,8))
sns.violinplot(x = data['Loudness..dB..'], y = data['Popularity'], split = True, inner = 'quart')
plt.show()
sns.jointplot(x = data.loc[:, 'Energy'], y = data.loc[:, 'Popularity'], size = 10, kind = 'regg')
G = sns.pairplot(data)
G.map_lower(sns.regplot)
plt.plot()
plt.show()