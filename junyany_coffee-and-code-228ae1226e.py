# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy import stats



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dat = pd.read_csv('../input/coffee-and-code/CoffeeAndCodeLT2018.csv')
dat.head()
dat['CodingWithoutCoffee'].value_counts()
N1= 30

N2 = 19
dat['CodingHours'].mean()
group_A_mean= dat[dat['CodingWithoutCoffee'] == 'Yes'].CodingHours.mean()

var_A = dat[dat['CodingWithoutCoffee'] == 'Yes'].CodingHours.var()

print(group_A_mean)

print(var_A)
group_B_mean = dat[dat['CodingWithoutCoffee'] == 'No'].CodingHours.mean()

var_B = dat[dat['CodingWithoutCoffee'] == 'No'].CodingHours.var()

print(group_B_mean)

print(var_B)
# standard deviation

s = np.sqrt((var_A + var_B)/2)

s
#calculate t statistics

t = (group_A_mean - group_B_mean)/(np.sqrt(var_A/N1 + var_B/N2))

t
df = N1+N2 -2
p = stats.t.sf(np.abs(t), N1+N2-2)*2

print("t = " + str(t))

print("p = " + str(2*p))
t2, p2= stats.ttest_ind_from_stats(group_A_mean, np.sqrt(var_A), N1, group_B_mean, np.sqrt(var_B), N2, equal_var = False)

print("t2 = " + str(t2))

print("p2 = " + str(2*p2))