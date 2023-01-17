# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import warnings

warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.
file = '../input/AllResults.csv'

data = pd.read_csv(file)
data.head()
data.info()
data.fillna(np.mean(data),inplace = True)

data.info()
data.dropna(inplace = True)
data.info()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

plt.figure(figsize = (12,10))

sns.heatmap(data.corr(),annot = True,cmap = 'coolwarm',lw = 0.2,linecolor =  'gold')
data.head()
sns.countplot(x = 'Ma',data = data,orient = 'v')
sns.countplot(x = 'HML',data = data)
sns.countplot(x = 'Re',data = data)
sns.distplot(data['PPID'],bins = 50,color = 'green')
sns.distplot(data['MathsDiff'],bins = 30,color = 'black',)
sns.distplot(data['EngEst'],bins = 30,color = 'red')
sns.jointplot(x ='MathsDiff',y = 'EngEst',data = data,kind = 'hex',color = 'green') 
sns.jointplot(x ='MathsDiff',y = 'MathsAct',data = data,kind = 'hex',color = 'gold') 
sns.jointplot(x ='EngEst',y = 'EngAct',data = data,kind = 'kde',color = 'black') 
sns.jointplot(x ='EbaccAct',y = 'EbaccDiff',data = data,color = 'teal') 
plt.figure(figsize = (12,10))

sns.boxplot(x = 'HML',y = 'MathsAct',data = data,hue = 'Re')