

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats    
import scipy.special as sps
%matplotlib inline
# Read in the data using the first column as the index
df = pd.read_csv('../input/train.csv', index_col=0)
df.head()
df.describe()








nb_total = df.place_id.count()
nb_unique = df.place_id.drop_duplicates().count()

print('Number place_ids: {}'.format(nb_total))
print('Unique place_ids: {}'.format(nb_unique))
print("Average number of duplicates: %.1f" % (nb_total/nb_unique))
from pandas.tools.plotting import scatter_matrix
df_sample = df[df.place_id == 4823777529]
scatter_matrix(df_sample.drop('place_id', axis=1), diagonal='kde', figsize=(11,11))
shape, scale = 1.34, 199.38 # mean and dispersion

s = df.place_id.value_counts().values

#Display the histogram of the samples, along with the probability density function:
ax = df.place_id.value_counts().plot.kde()
ax.set_xlim(0, 2000)

count, bins, ignored = plt.hist(s, 100, normed=True)
#y = bins**(shape-1)*(np.exp(-bins/scale) /
#                      (sps.gamma(shape)*scale**shape))
#y = stats.gamma.pdf(bins, a=.6, loc=.999, scale=2.0549)
#rv = stats.maxwell(loc=-249.6547, scale=336.860199)
rv = stats.frechet_r(1.1, loc=0.89, scale=280)
y = rv.pdf(bins)
ax.plot(bins, y, linewidth=2, color='r')
