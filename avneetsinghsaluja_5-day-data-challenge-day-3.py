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
import pandas as pd

import matplotlib.pyplot as plt



cereal_data = pd.read_csv('../input/80-cereals/cereal.csv')

cereal_data.head(10)
cereal_protein = cereal_data['protein']

plt.title('Protein Data for cereals')

plt.xlabel('Quantity of protein')

plt.ylabel('Number of cereals')

plt.hist(cereal_data['protein'], edgecolor='black')
from scipy.stats import ttest_ind #for performing t-test

from scipy.stats import probplot # for a qqplot

import pylab

probplot(cereal_data["sodium"], dist="norm", plot=pylab)

hotCereals_sodium = cereal_data['sodium'][cereal_data['type']=='H']

coldCereals_sodium = cereal_data['sodium'][cereal_data['type']=='C']

ttest_ind(hotCereals_sodium, coldCereals_sodium, equal_var=False)



#When pvalue is less than alpha (0.05), we can reject null hypothesis which states that there is not much 

# difference between the two distributions

# => there is significant difference between the distribution of value of sodium for cold and hot cereals
print(hotCereals_sodium.mean())

print(coldCereals_sodium.mean())
plt.hist(coldCereals_sodium, alpha = 0.5, label = 'cold')

plt.hist(hotCereals_sodium, label = 'hot')

plt.title('Sodium content of cereals (by type)')