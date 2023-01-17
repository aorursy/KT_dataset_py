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
# take a look at the data
kickstarters_2017.head()
kickstarters_2017.info()

# we know there are 15 columns in the dataset and there are 378661 rows.
kickstarters_2017.isnull().any()
# we see 'name' and 'usd pledged' has missing values
kickstarters_2017.isnull().sum()

# there are 4 missing values in name and 3797 values in 'usd pledged'
# how many total missing values do we have?
total_cells = np.product(kickstarters_2017.shape)
total_missing = (kickstarters_2017.isnull().sum()).sum()

# percent of data that is missing
total_missing / total_cells

# good! not many missing data
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
# select the goal column 
goal = kickstarters_2017.goal

# scale the goal from 0 to 1
scaled_goal = minmax_scaling(goal, columns=[0])

# plot the original & scaled data together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(goal, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_goal, ax=ax[1])
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
# We looked at the usd_pledged_real column. What about the "pledged" column? Does it have the same info?
# take a look at the data
kickstarters_2017.pledged.head()

# get only positive pledges (using their indexes)
positive_pledged = kickstarters_2017.pledged[kickstarters_2017.pledged >0]

# normalize the pledges (w/ Box-Cox)
normalized_pledged = stats.boxcox(positive_pledged)[0]

# plot both dataset together
fit, ax = plt.subplots(1,2)

sns.distplot(positive_pledged, ax = ax[0])
ax[0].set_title('Original Data')
sns.distplot(normalized_pledged, ax = ax[1])
ax[1].set_title('Normalized Data')
