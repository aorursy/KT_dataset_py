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
# get a bird's-eye view of our data 
print(kickstarters_2017.head())
# set seed for reproducibility
np.random.seed(0)
# generate 1000 data points randomly drawn from an exponential distribution
original_data = np.random.exponential(size = 10)
print(original_data)
# mix-max scale the data between 0 and 1
scaled_data = minmax_scaling(original_data, columns = [0])
print(scaled_data)
# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")
# normalize the exponential data with boxcox
normalized_data = stats.boxcox(original_data)
print(normalized_data)
# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_data[0], ax=ax[1])
ax[1].set_title("Normalized data")
# select the usd_goal_real column
usd_goal = kickstarters_2017.usd_goal_real
#print(usd_goal)
# scale the goals from 0 to 1
scaled_data = minmax_scaling(usd_goal, columns = [0])
#print(scaled_data)
# plot the original & scaled data together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(kickstarters_2017.usd_goal_real, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")
# Your turn! 

# We just scaled the "usd_goal_real" column. What about the "goal" column?
# select the goal column
goal = kickstarters_2017.goal
# scale the goals from 0 to 1
scaled_goal = minmax_scaling(goal, columns = [0])
# plot the original $ scaled data together to compare
my_fig, my_ax = plt.subplots(1, 2)
sns.distplot(kickstarters_2017.goal, ax=my_ax[0])
my_ax[0].set_title("Original Goal Data")
sns.distplot(scaled_goal, ax=my_ax[1])
my_ax[1].set_title("Scaled Goal Data")

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
index_of_positive_data = kickstarters_2017.pledged > 0

# get only positive pledges (using their indexes)
positive_data = kickstarters_2017.pledged.loc[index_of_positive_data]

# normalize the pledges (w/ Box-Cox)
normalized_data = stats.boxcox(positive_data)[0]

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(positive_data, ax=ax[0])
ax[0].set_title("Original Pledged Data")
sns.distplot(normalized_data, ax=ax[1])
ax[1].set_title("Normalized Pledged Data")