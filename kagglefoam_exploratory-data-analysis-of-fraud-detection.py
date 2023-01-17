# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import datasets

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df1 = pd.read_csv('../input/PS_20174392719_1491204439457_log.csv')
df1.info()
df1.describe()
df1.head(5)
print(df1.type.value_counts())
# The column ‘type’ is categorical,  the frequency distribution for each category.

df1['type'].value_counts()
import statsmodels.api as sm

# Correlation Matrix

# The correlation function uses Pearson correlation coefficient, which results in a number between -1 to 1

# A strong negative relationship is indicated by a coefficient closer to -1 and a strong positive correlation is indicated by a coefficient toward 1

corr = df1.corr()

print(corr)
sm.graphics.plot_corr(corr, xnames=list(corr.columns))

plt.show()
df2 = pd.DataFrame(df1, columns = ['amount', 'newbalanceDest'])
df2.plot(kind='scatter', x='amount', y='newbalanceDest', title='Amount vs New Balance Destination')
df2.corr()