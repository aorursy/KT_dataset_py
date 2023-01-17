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
import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()

%matplotlib inline
CC = pd.read_csv("../input/creditcardfraud/creditcard.csv")

CC.head(5)
CC.shape
CC.columns
print(CC.describe())
pd.isnull(CC).isnull().sum()
CC_ = CC.sample(frac = 0.1, random_state = 1)

print(CC_)
print(CC_.shape)
CC_.hist(figsize = (20, 20))

plt.show()
Fraud = CC_[CC_['Class'] == 0]

Valid = CC_[CC_['Class'] == 1] 
print('Fraud Cases: {}'.format(len(Fraud)))

print('Valid Cases: {}'.format(len(Valid)))
# Calculating the percentage of outliers(noise)



fraction_outliers = len(Valid) / float(len(Fraud))

print(fraction_outliers)
# Applying Correlation_matrix 



Corr_matrix = CC_.corr()



fig = plt.figure(figsize = (18, 18))

sns.heatmap(Corr_matrix, vmax = 0.5, square = True)

plt.show()