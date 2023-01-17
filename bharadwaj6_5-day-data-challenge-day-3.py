import numpy as np

import pandas as pd

from scipy.stats import ttest_ind



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
cereal = pd.read_csv('../input/cereal.csv')

cereal.info()
cereal.head().T
cereal.describe()
cereal['calories'].hist()
from scipy.stats import probplot

import pylab
probplot(cereal['sodium'], dist='norm', plot=pylab)

pylab.show()
cereal['sodium'].hist()
hot_cereal = cereal[cereal['type'] == 'H']

cold_cereal = cereal[cereal['type'] == 'C']
hot_cereal.head()
cold_cereal.head()
ttest_ind(hot_cereal['sodium'], cold_cereal['sodium'], equal_var=False)
hot_cereal['sodium'].hist()
cold_cereal['sodium'].hist()
cereal['rating'].hist()
good_rated = cereal[cereal['rating'] > 50]

poor_rated = cereal[cereal['rating'] <= 50]
print('no of good rated are: {} and poor rated ones are: {}'.format(good_rated.shape[0], poor_rated.shape[0]))
good_rated.describe()
poor_rated.describe()
ttest_ind(good_rated['vitamins'], poor_rated['vitamins'], equal_var=False)
good_rated['vitamins'].mean()
poor_rated['vitamins'].mean()