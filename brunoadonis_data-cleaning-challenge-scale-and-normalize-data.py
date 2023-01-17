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
# Your turn! 
# select the goal column
goal_ = kickstarters_2017.goal
# We just scaled the "usd_goal_real" column. What about the "goal" column?
# scale goals from 0 to 1
scaled_data_goal_ = minmax_scaling(goal_, columns = [0])
# plot the original & scaled data together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(goal_, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data_goal_, ax=ax[1])
ax[1].set_title("Scaled data")
# Your turn! 
usd_goal_real_ = kickstarters_2017.usd_goal_real
# We just scaled the "usd_goal_real" column. What about the "goal" column?
scaled_data_usd_goal_real_ = minmax_scaling(usd_goal_real_, columns = [0])

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(usd_goal_real_, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data_usd_goal_real_, ax=ax[1])
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
index_of_positive_pledged_ = kickstarters_2017.pledged > 0

# get only positive pledges (using their indexes)
positive_pledged_ = kickstarters_2017.usd_pledged_real.loc[index_of_positive_pledged_]

# normalize the pledges (w/ Box-Cox)
normalized_pledged_ = stats.boxcox(positive_pledged_)[0]

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(positive_pledged_, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_pledged_, ax=ax[1])
ax[1].set_title("Normalized data")
kickstarters_2017.head()
print("Original Data")
print("pledged :: Max: {}, Min: {}, Mean: {}, Median: {} ".format(kickstarters_2017.pledged.max(), kickstarters_2017.pledged.min(), kickstarters_2017.pledged.mean(), kickstarters_2017.pledged.median()))
print("usd_pledged_real :: Max: {}, Min: {}, Mean: {}, Median: {} ".format(kickstarters_2017.usd_pledged_real.max(), kickstarters_2017.usd_pledged_real.min(), kickstarters_2017.usd_pledged_real.mean(), kickstarters_2017.usd_pledged_real.median()))
print("usd_goal_real :: Max: {}, Min: {}, Mean: {}, Median: {} ".format(kickstarters_2017.usd_goal_real.max(), kickstarters_2017.usd_goal_real.min(), kickstarters_2017.usd_goal_real.mean(), kickstarters_2017.usd_goal_real.median()))

print("Normalized")
print("pledged :: Max: {}, Min: {}, Mean: {}, Median: {} ".format(max(normalized_pledged_), min(normalized_pledged_), np.mean(normalized_pledged_), np.median(normalized_pledged_)))
print("usd_pledged_real :: Max: {}, Min: {}, Mean: {}, Median: {} ".format(max(normalized_pledges), min(normalized_pledges), np.mean(normalized_pledges), np.median(normalized_pledges)))
print("usd_goal_real :: Max: {}, Min: {}, Mean: {}, Median: {} ".format(max(scaled_data_usd_goal_real_), min(scaled_data_usd_goal_real_), np.mean(scaled_data_usd_goal_real_), np.median(scaled_data_usd_goal_real_)))
