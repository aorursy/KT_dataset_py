# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#fetching and rading the csv file

dir = '../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv'

ny = pd.read_csv(dir)

ny.head()
from scipy import stats

stats.zscore(ny.price)

import numpy as np

mean = np.mean(ny.price)

std = np.std(ny.price)



def z_score(value, mean, std):

    return (value-mean)/ std
import random

values = []

#randomly select values from the price column

for i in list(range(0, 5)):

    value = random.choice(ny.price)

    values.append(value)

    

    

print(values)
for val in values:

    z = z_score(val, mean, std)

    print(z)
import matplotlib.pyplot as plt

%matplotlib inline



plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})

plt.hist(ny.price, bins=100, range=(0, 1000))

plt.show()
print(mean)

print(std)