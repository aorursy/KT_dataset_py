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

# so what does the data itself look like?  Remember, this is randomly generated
small_original = original_data[0:10]
print(small_original)
#scaled_data.  See that the two vectors are not the same dimension.  Why?
small_scale = scaled_data[0:10].transpose()
print(small_scale)
#ratio?  Should be the same!
print(small_original / small_scale)
# normalize the exponential data with boxcox
normalized_data = stats.boxcox(original_data)
# what is the shape of this
print(np.shape(normalized_data))
# so what is the shape of things stored in this tuple?
print(np.shape(normalized_data[0]))
print(np.shape(normalized_data[1]))
# and then what do the values look like?
print(normalized_data[0][0:10])
print(normalized_data[1])
# and what are the extremes of the values?
print("Max: " + str(max(normalized_data[0])))
print("Min: " + str(min(normalized_data[0])))
# normalize the exponential data with boxcox
normalized_data = stats.boxcox(original_data)

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_data[0], ax=ax[1])
ax[1].set_title("Normalized data")

# so what does the data itself look like?  Remember, this is randomly generated
small_original = original_data[0:10]
print(small_original)
#normalized_data.  See that the two vectors are not the same dimension again.  All data is in [-x,x] ideally.
small_normalized = normalized_data[0][0:10]
print(small_normalized)
#ratio?  Should not be the same!
print(small_original / small_normalized)
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
# select the usd_goal_real column
goal = kickstarters_2017.goal

# scale the goals from 0 to 1
scaled_data = minmax_scaling(goal, columns = [0])

# plot the original & scaled data together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(goal, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
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
index_of_positive_pledges = kickstarters_2017.pledged > 0

# get only positive pledges (using their indexes)
positive_pledges = kickstarters_2017.pledged.loc[index_of_positive_pledges]

# normalize the pledges (w/ Box-Cox)
normalized_pledges = stats.boxcox(positive_pledges)[0]

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(positive_pledges, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_pledges, ax=ax[1])
ax[1].set_title("Normalized data")