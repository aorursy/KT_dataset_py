# modules we'll use
import pandas
import numpy

# for Box-Cox Transformation
from scipy import stats

# for min_max scaling
from mlxtend.preprocessing import minmax_scaling

# plotting modules
import seaborn as sns
import matplotlib.pyplot as plt

# read in all our data
kickstarters_2017 = pandas.read_csv("../input/kickstarter-projects/ks-projects-201801.csv")

# set seed for reproducibility
numpy.random.seed(0)

# Looking at the data
kickstarters_2017.info()
sample_data = kickstarters_2017.sample(5)
sample_data
# Checking if the two columns are same or not: 'usd pledged' and 'usd_pledged_real'
kickstarters_2017['usd pledged'].equals(kickstarters_2017['usd_pledged_real'])
# inference: columns are not same
# generate 1000 data points randomly drawn from an exponential distribution
original_data = numpy.random.exponential(size = 1000)
print(type(original_data))
print(original_data[0: 10])
# mix-max scale the data between 0 and 1
scaled_data = minmax_scaling(original_data, columns = [0])
print(scaled_data[0:10])
# plot both together to compare
fig, ax = plt.subplots(1,2)
sns.distplot(original_data, ax = ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax = ax[1])
ax[1].set_title("Scaled data")
# normalize the exponential data with boxcox
normalized_data = stats.boxcox(original_data)
print(type(normalized_data))
normalized_data
# plot both together to compare
fig, ax = plt.subplots(1,2)
sns.distplot(original_data, ax = ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_data[0], ax = ax[1])
ax[1].set_title("Normalized data")
# select the usd_goal_real column
usd_goal = kickstarters_2017.usd_goal_real
print(usd_goal[0:10])

print("-----------------")

# scale the goals from 0 to 1
scaled_data = minmax_scaling(usd_goal, columns = [0])
print(scaled_data[0:10])
# plot the original & scaled data together to compare
fig, ax = plt.subplots(1,2)
sns.distplot(kickstarters_2017.usd_goal_real, ax = ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax = ax[1])
ax[1].set_title("Scaled data")
# Your turn! 

# We just scaled the "usd_goal_real" column. What about the "goal" column?
# selecting the goal column
goal = kickstarters_2017.goal
print(goal[0:10])

print("-----------------------")

# scaling the column
scaled_goal = minmax_scaling(goal, columns = [0])
print(scaled_goal[0:10])

# plot the original and scaled column
fig, ax = plt.subplots(1, 2)
sns.distplot(kickstarters_2017.goal, ax = ax[0])
ax[0].set_title("Original_data")
sns.distplot(scaled_goal, ax = ax[1])
ax[1].set_title("Scaled_data")
# get the index of all positive pledges (Box-Cox only takes postive values)
index_of_positive_pledges = kickstarters_2017.usd_pledged_real > 0
print(index_of_positive_pledges[0:5])

# get only positive pledges (using their indexes)
positive_pledges = kickstarters_2017.usd_pledged_real.loc[index_of_positive_pledges]
print(positive_pledges[0:10])

# normalize the pledges (w/ Box-Cox)
normalized_pledges = stats.boxcox(positive_pledges)[0]
print(normalized_pledges[0:10])
# plot both together to compare
fig, ax = plt.subplots(1,2)
sns.distplot(positive_pledges, ax = ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_pledges, ax = ax[1])
ax[1].set_title("Normalized data")
# Your turn! 
# We looked as the usd_pledged_real column. What about the "pledged" column? Does it have the same info?
# Get the index of all positive pledged
ind_positive_pledged = kickstarters_2017.pledged > 0
print(ind_positive_pledged[0:5])

# Getting the pledged values
pledged_values = kickstarters_2017.pledged.loc[ind_positive_pledged]
print(pledged_values[0:10])

# Normalizing the pledged values
normalized_pledged = stats.boxcox(pledged_values)[0]
print(normalized_pledged[0:10])
# plotting both the variables
fig, ax = plt.subplots(1, 2)
sns.distplot(pledged_values, ax = ax[0])
ax[0].set_title("Original_data")
sns.distplot(normalized_pledged, ax = ax[1])
ax[1].set_title("Normalized_data")