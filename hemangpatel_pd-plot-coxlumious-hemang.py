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
lab_names = ['AAAA', 'BBBB', 'CCCC']

lab_values = [10, 30, 20]

data =     {
    'lab': lab_names, 
    'val': lab_values
}

df = pd.DataFrame(data)

ax = df.plot.bar(x='lab', y='val', rot=45)
mu, sigma = 0, 0.1 # mean and standard deviation
mu = 5 # mean and standard deviation
sigma = 1 # mean and standard deviation

s = np.random.normal(mu, sigma, 10000)
# Verify the mean and the variance:

abs(mu - np.mean(s)) < 0.01
abs(sigma - np.std(s, ddof=1)) < 0.01
# Display the histogram of the samples, along with the probability density function:

import matplotlib.pyplot as plt
count, bins, ignored = plt.hist(s, 30, normed=True)
plt.plot(bins , 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
plt.show()