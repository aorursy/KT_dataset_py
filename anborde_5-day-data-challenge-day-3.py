# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from scipy.stats import ttest_ind

from scipy.stats.mstats import normaltest

from scipy.stats import probplot 

import pylab

import seaborn as sns
train = pd.read_csv("../input/train.csv", na_values=['NaN'])



print(train.columns.values)



train.tail()

train = train.dropna()
# Printing description of the columns

train.describe()


train.describe(include=['O'])
probplot(train['Age'].dropna(), dist="norm", plot=pylab)
survived = train['Age'][train['Survived'] == 1]

not_survived = train['Age'][train['Survived'] == 0]
print(ttest_ind(survived, not_survived, equal_var=False,nan_policy='omit'))

import matplotlib.pylab as plt



sns.distplot(survived.dropna(), label='Survived')

sns.distplot(not_survived.dropna(), label='Not Survived')

plt.legend()