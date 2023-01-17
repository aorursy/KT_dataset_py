import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import ttest_ind
# read the dataset

data = pd.read_csv('../input/museums.csv')



# summarize the dataset

data.describe()
# list all different `Museum Type`

data['Museum Type'].value_counts()
# split the data into zoo and other museums

zoos = data[data['Museum Type'] == 'ZOO, AQUARIUM, OR WILDLIFE CONSERVATION']

museums = data[data['Museum Type'] != 'ZOO, AQUARIUM, OR WILDLIFE CONSERVATION']



zoosRevenue = zoos['Revenue'].dropna().drop_duplicates()

museumsRevenue = museums['Revenue'].dropna().drop_duplicates()



# apply log normalization

zoosRevenue = zoosRevenue.apply(np.log)

museumsRevenue = museumsRevenue.apply(np.log)



zoosRevenue = zoosRevenue[zoosRevenue > 0]

museumsRevenue = museumsRevenue[museumsRevenue > 0]



zoosRevenue.head()
# perform t-test on revenue

ttest_ind(zoosRevenue, zoosRevenue, equal_var=False)
plt.figure(figsize=(14, 7))

sns.distplot(zoosRevenue)

sns.distplot(museumsRevenue)