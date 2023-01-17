# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from scipy import stats

sales=pd.read_csv('../input/sales.csv')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory









# Any results you write to the current directory are saved as output.
plt.hist(sales['Phones'],color='teal')

plt.title('Histogram of Sales of Phones')

plt.xlabel('Phones')

plt.ylabel('Sales Frequency')

plt.show()
plt.hist(sales['Tvs'],color='turquoise')

plt.title('Histogram of Sales of Tvs')

plt.xlabel('Tvs')

plt.ylabel('Sales Frequency')

plt.show()


sales.describe()
ttest=stats.ttest_ind(sales['Phones'],sales['Tvs'])

print ('t-test independent', ttest)