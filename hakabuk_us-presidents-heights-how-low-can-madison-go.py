import warnings

warnings.filterwarnings("ignore")

import pandas as pd 

import numpy as np
data = pd.read_csv('../input/president_heights_new.csv', index_col='order')

print("The shape of the data is {0} rows and {1} columns.".format(data.shape[0], data.shape[1]))
data.tail()
data['height(cm)'].describe()
# who is the shortest US president?

data.loc[data['height(cm)'].idxmin()][0]
# who is the tallest?

# data.loc[data['height(cm)'].idxmax()]

# since this will return only one value, we'll use a mask: 

mask = data['height(cm)'] == 193

data.name[mask]
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns 

sns.set(style='white', color_codes=True); # set plot style
plt.hist(data['height(cm)'], alpha=0.6)

plt.title('Height Distribution of US Presidents')

plt.xlabel('height (cm)')

plt.ylabel('number')

sns.despine()