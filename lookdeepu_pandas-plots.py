

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt
%matplotlib inline
plt.plot([1,2,3,4]) #test code
phony = pd.read_csv('../input/bigml_59c28831336c6604c800002a.csv')
phony.head()
phony.columns
phony.describe()
phony.boxplot(input(),rot=45, fontsize=15, grid = False)
phony.corr()
phony.hist('account length', grid = False)
import seaborn as sns

# Density Plot and Histogram of all arrival delays

sns.distplot(phony['account length'], hist=True, kde=True, 

             bins=int(180/5), color = 'darkblue', 

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 6})
phony.shape
data = pd.read_csv(input())