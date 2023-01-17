import pandas as pd
import numpy as np

df = pd.read_csv('../input/nflplaybyplay2009to2016/NFL Play by Play 2009-2017 (v4).csv')
df.head()
missing_values_count = df.isnull().sum()
missing_values_count
total_cells = np.product(df.shape)
total_missing = missing_values_count.sum()
percent_missing = (total_missing/total_cells) * 100
print(percent_missing)
missing_values_count[0:10]
columns_with_na_dropped = df.dropna(axis=1)
columns_with_na_dropped.head()
print("Columns in orginal DF: %d" % df.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
subset_df = df.loc[:, 'EPA':'Season'].head()
subset_df
subset_df.fillna(0)
subset_df.fillna(method='bfill', axis=0).fillna(0)
# for Box-Cox Transformation
from scipy import stats

# for min_max scaling
from mlxtend.preprocessing import minmax_scaling

# plotting modules
import seaborn as sns
import matplotlib.pyplot as plt

# set seed for reproducibility
np.random.seed(0)
# generate 1000 data points randomly drawn from an exponential distribution
original_data = np.random.exponential(size=1000)

# mix-max scale the data between 0 and 1
scaled_data = minmax_scaling(original_data, columns=[0])

# plot both together to compare
fig, ax = plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")