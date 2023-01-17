# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import scipy as scipy

from scipy import stats



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
hosp = pd.read_csv('../input/mimic3d.csv')

# getting general information

hosp.info()



hosp.describe()
hosp.dtypes

hosp.shape
hosp.head(20)
hosp.tail(20)
hosp['AdmitDiagnosis'].unique().shape
hosp['age'].unique().shape
scipy.stats.describe(hosp.age)
scipy.stats.kurtosis(hosp.age)
hosp.isnull().sum()
plt.figure(figsize=(20,10))

sns.countplot(x="age", data=hosp, palette="bwr")

plt.title('Distibution of Age')

plt.xticks(rotation=90)

plt.show()
#corelation matrix

plt.figure(figsize=(20,20))

sns.heatmap(cbar=False,annot=True,data=hosp.corr()*100,cmap='coolwarm')

plt.title('% Corelation Matrix')

plt.show()
scipy.stats.chisquare(hosp.age)
scipy.stats.pearsonr(hosp.age,hosp.NumRx)
scipy.stats.skew(hosp.age, axis=0, bias=True, nan_policy='propagate')
from scipy import stats

stats.describe(hosp.age)
np.histogram(hosp.age, bins=40)

plt.show()
import matplotlib.pyplot as plt

plt.figure(figsize=(20,20))



n, bins, patches = plt.hist(x=hosp.age, bins='auto', color='#0504aa',

                            alpha=0.7, rwidth=0.85)

plt.grid(axis='y', alpha=0.75)

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.title('Age Distribution')

plt.text(23, 45, r'$\mu=15, b=3$')

maxfreq = n.max()



#plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)