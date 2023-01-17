# modules we'll use
import pandas as pd
import numpy as np

# for Box-Cox Transformation
from scipy import stats

# for min_max scaling
from mlxtend.preprocessing import minmax_scaling

# plotting modules
import seaborn as sns
import matplotlib.pyplot as plt

# read in all our data
kickstarters_2017 = pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv")

# set seed for reproducibility
np.random.seed(0)
# generate 1000 data points randomly drawn from an exponential distribution
original_data = np.random.exponential(size = 1000)

# mix-max scale the data between 0 and 1
scaled_data = minmax_scaling(original_data, columns = [0])

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")
# normalize the exponential data with boxcox
normalized_data = stats.boxcox(original_data)

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_data[0], ax=ax[1])
ax[1].set_title("Normalized data")
# select the usd_goal_real column
usd_goal = kickstarters_2017.usd_goal_real

# scale the goals from 0 to 1
scaled_data = minmax_scaling(usd_goal, columns = [0])

# plot the original & scaled data together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(kickstarters_2017.usd_goal_real, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")
# Your turn! 

# We just scaled the "usd_goal_real" column. What about the "goal" column?
# select the goal
goal = kickstarters_2017.goal

# scale the goals from 0 to 1
scaled_data_goal = minmax_scaling(goal, columns = [0])

# plot the original & scaled data together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(kickstarters_2017.goal, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data_goal, ax=ax[1])
ax[1].set_title("Scaled data")

# get the index of all positive pledges (Box-Cox only takes postive values)
index_of_positive_pledges = kickstarters_2017.usd_pledged_real > 0

# get only positive pledges (using their indexes)
positive_pledges = kickstarters_2017.usd_pledged_real.loc[index_of_positive_pledges]

# normalize the pledges (w/ Box-Cox)
normalized_pledges = stats.boxcox(positive_pledges)[0]

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(positive_pledges, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_pledges, ax=ax[1])
ax[1].set_title("Normalized data")
# Your turn! 
# We looked as the usd_pledged_real column. What about the "pledged" column? Does it have the same info?

# get the index of all positive pledges (Box-Cox only takes postive values)
index_of_positive_pledges_2 = kickstarters_2017.pledged > 0

# get only positive pledges (using their indexes)
positive_pledges_2 = kickstarters_2017.pledged.loc[index_of_positive_pledges_2]

# normalize the pledges (w/ Box-Cox)
normalized_pledges_2 = stats.boxcox(positive_pledges_2)[0]

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(positive_pledges_2, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_pledges_2, ax=ax[1])
ax[1].set_title("Normalized data")
import pandas as pd
# reading the dataset
calcofi = pd.read_csv("../input/calcofi/bottle.csv")
calcofi.tail()
calcofi_subset = calcofi.loc[:,'Depthm':'O2Satq']
calcofi_subset.tail()

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# sns.pairplot(calcofi_subset, x_vars = 'Depthm', y_vars = 'O2ml_L')
sns.jointplot('Depthm','O2ml_L',calcofi_subset,  kind="kde", size=7, space=0)
calcofi_subset.isnull().sum()
# Let's compare water temperature using fillna() vs mean()
fig, ax=plt.subplots(1,2)
sns.distplot(calcofi.T_degC.dropna(), ax=ax[0])
ax[0].set_title("Dropna")
sns.distplot(calcofi.T_degC.fillna(calcofi.T_degC.mean()), ax=ax[1])
ax[1].set_title("Fillna with mean")
# Let's compare oxygen saturation using fillna() vs mean()
fig, ax=plt.subplots(1,2)
sns.distplot(calcofi.O2Satq.dropna(), ax=ax[0])
ax[0].set_title("Dropna")
sns.distplot(calcofi.O2Satq.fillna(calcofi.O2Satq.mean()), ax=ax[1])
ax[1].set_title("Fillna mean")
# normalize oxygen sat (w/ Box-Cox)
normalized_ox_dropna = stats.boxcox(calcofi.O2Satq.dropna())[0]
# normalize oxygen sat (w/ Box-Cox)
normalized_ox_fillna = stats.boxcox(calcofi.O2Satq.fillna(calcofi.O2Satq.mean()))[0]

fig, ax=plt.subplots(1,2)
sns.distplot(normalized_ox_dropna, ax=ax[0])
ax[0].set_title("Norm Dropna")
sns.distplot(normalized_ox_fillna, ax=ax[1])
ax[1].set_title("Norm Fillna mean")
# plotting the distributions part 1
fig, ax=plt.subplots(1,2)
sns.distplot(calcofi_mean.T_degC, ax=ax[0])
ax[0].set_title("Water Temperature")
sns.distplot(calcofi_mean.Salnty, ax=ax[1])
ax[1].set_title("Salinity")
# plotting the distributions part 2
fig, ax=plt.subplots(1,2)
sns.distplot(calcofi_mean.O2Satq, ax=ax[0])
ax[0].set_title("Oxygen Saturation")
sns.distplot(calcofi_mean.Depthm, ax=ax[1])
ax[1].set_title("Depth")