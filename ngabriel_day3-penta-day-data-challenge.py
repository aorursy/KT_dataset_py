# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.stats import ttest_ind

import seaborn as sns

from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

data=pd.read_csv("../input/cereal.csv")

#data.describe



print(ttest_ind(data['carbo'],data['fiber'],equal_var=False))

fig,axs=plt.subplots(nrows=1,ncols=2,figsize=(17,5))



fig.subplots_adjust(wspace=2)

fig.suptitle("Histogram of carbo and fiber".upper())

ax=sns.distplot(data['carbo'],kde=False,ax=axs[0],color='black',hist_kws=dict(edgecolor='red'))

ax.set(xlabel='Range of Carbo in diet',ylabel='Frequency of carbo')

ax=sns.distplot(data['fiber'],kde=False,ax=axs[1],color='green',hist_kws=dict(edgecolor='black'))

ax.set(xlabel='Range of Fiber in diet',ylabel='Frequency of Fiber')
