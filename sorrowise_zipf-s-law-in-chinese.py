# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Import the required modules and set some parameters

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
sns.set(style="white",color_codes=True)
plt.rcParams['figure.figsize'] = (15,9.27)
# import data with pandas
df = pd.read_csv('../input/hanzidb/hanziDB.csv')
# View basic information of the data
df.head(10)
# Let us look at the relationship between the number of strokes in Chinese characters and the frequency of their use. 
# Intuitively, we believe that the more counts of strokes, the lower the frequency of use should be.

# This stroke data was incorrectly entered as "8 9" and needs to be corrected.
df.loc[804,'stroke_count'] = 8

# Convert data types to integers
df.frequency_rank = df.frequency_rank.astype(int)
df.stroke_count = df.stroke_count.astype(int)
# Observe the basic information of Chinese stroke data
df.stroke_count.describe()
# Let us look at the relationship between the number of strokes in Chinese characters 
# and the frequency of their use with boxplot from seaborn library

sns.boxplot(df.stroke_count,df.frequency_rank)
hanzi = pd.read_excel('../input/chinese-character-frequency-statistics/hanzi_freq.xlsx')
hanzi.head(10)
plt.plot(hanzi.cum_freq_percent,linewidth=3)
plt.xlabel('Rank according to frequency of use')
plt.ylabel('Cumulative frequency')
plt.xlim(0,10000)
plt.ylim(0,100)
plt.scatter(hanzi['rank'][:150],hanzi.freq_percent[:150])
plt.xlim(0,150)
plt.ylim(0,5)
plt.xlabel('Rank according to frequency of use')
plt.ylabel('The percentage frequency of use')
hanzi['log_rank'] = np.log10(hanzi['rank'])
hanzi['log_freq'] = np.log10(hanzi['freq'])
hanzi.head(10)
sns.regplot(hanzi.log_rank,hanzi.log_freq)
plt.text(1.5,12,r'$log(freq)=14.13-3.32log(rank),\ R^2=0.812$',fontsize=20)
def reg(y,*args):
    import statsmodels.api as sm
    x = np.vstack((args)).T
    mat_x = sm.add_constant(x)
    res = sm.OLS(y,mat_x).fit()
    print(res.summary())
reg(hanzi.log_freq,hanzi.log_rank)
sns.regplot(hanzi.log_rank[:200],hanzi.log_freq[:200])
x = np.linspace(0,3)
y = -x + 7
plt.plot(x,y,'r--')
plt.text(0.75,6.8,r'$log(freq)=6.81-0.63log(rank),\ R^2=0.991$',fontsize=20)
plt.xlim(0,2.5)
plt.ylim(5.3,7)
reg(hanzi.log_freq[:200],hanzi.log_rank[:200])