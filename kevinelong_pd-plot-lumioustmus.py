# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
%matplotlib inline
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
lab_names = ['AAAA', 'BBBB', 'CCCC', 'DDDD']

lab_values = [10, 30, 20, 44]

data = {
    'XXXX': lab_names, 
    'YYYY': lab_values
}

# from pandas import DataFrame
# df = DataFrame(data)

df = pd.DataFrame(data)

ax = df.plot.bar(x='XXXX', y='YYYY', rot=45)

#one liner
pd.DataFrame([20, 30, 20, 44]).plot.bar()
a, b, c = 1, 2, 3
print(c)
# np.mean(pd.DataFrame([12,13,15,11,98]))
# np.mean([12,13,15,11,98])
# Assign two variables on one line because you can do that in python
# mu, sigma = 0, 0.1 # mean and standard deviation

# sane person would assign one variable per line like this
mu = 0 # mean AKA center
sigma = 0.1 # standard deviation AKA scale
quantity = 1000 # times

#https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.normal.html
s = np.random.normal(mu, sigma, quantity)
# print(s)

# Verify the mean and the variance:
print("MEAN", abs(mu - np.mean(s)) < 0.01)
print("VARIANCE:", abs(sigma - np.std(s, ddof=1)) < 0.01)


# Display the histogram of the samples, along with the probability density function:
import matplotlib.pyplot as plt
count, bins, ignored = plt.hist(s, 30, density=True)
plt.plot(bins , 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
plt.show()