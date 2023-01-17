# Import libraries, including t-test libraries required for this challenge



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.stats import ttest_ind

import matplotlib.pyplot as plt



# Import data into a dataframework

museumsdf = pd.read_csv('../input/museums.csv', low_memory=False)
museumsdf.info()
museumsdf['Museum Name'].head(15)
museumsdf['Museum Type'].head(15)
zoos = museumsdf[museumsdf['Museum Type'].str.contains("ZOO", case=False)]
zoos.info()
zoo_with_revenue = zoos[zoos['Revenue'] > 0]
zoo_with_revenue.info()
others = museumsdf[museumsdf['Museum Type'].str.contains("ZOO")==False]
others.info()
others_with_revenue = others[others['Revenue']>0]
others_with_revenue.info()
ttest_ind(zoo_with_revenue['Revenue'], others_with_revenue['Revenue'], equal_var=False)
# Zoo Revenue Histogram

plt.title("Zoo Revenue Histogram")

plt.hist(zoo_with_revenue['Revenue'], bins=30)
# Other Type of Museums Revenue Histogram

plt.title("Other Type of Museums Revenue Histogram")

plt.hist(others_with_revenue['Revenue'], bins=30)