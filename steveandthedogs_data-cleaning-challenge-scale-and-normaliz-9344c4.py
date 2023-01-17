# modules we'll use
import pandas as pd
import numpy as np

# for Box-Cox Transformation
from scipy import stats

# for min_max scaling
# Steve's trouble... I've been running these exercises on my own computer's 
# python3 enviornment setup from Anaconda3
# I did not have this mlxtend library and had to install it via pip install mlxtend
# I had some trouble installing the first time, some sort of version error, 
# but I tried again it it all worked... enviornment and library install fun!
from mlxtend.preprocessing import minmax_scaling

# plotting modules
import seaborn as sns
import matplotlib.pyplot as plt

# read in all our data
kickstarters_2017 = pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv")

# set seed for reproducibility
np.random.seed(0)
kickstarters_2017.shape

kickstarters_2017.describe()
kickstarters_2017.info(verbose=True)
kickstarters_2017.head(5)
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

# select the usd_goal_real column
goal_s = kickstarters_2017.goal

# scale the goals from 0 to 1
scaled_data = minmax_scaling(goal_s, columns = [0])

# plot the original & scaled data together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(goal_s, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")

# ok at first I didn't think anything happened
# now I realize that the original data was in scientific notation
# 1e8 which is 100,000,000 the max was scaled to 1.00
goal_s.describe()
# get the index of all positive pledges (Box-Cox only takes postive values)
# this is a great python statement
# it will assign True or False to each 'row' based on the condition 
index_of_positive_pledges = kickstarters_2017.usd_pledged_real > 0

# get only positive pledges (using their indexes)
# this fun piece of code picks out all of the usd_pledged_real values ONLY IF they are positive,
# as found within the index_of_positive_pledges
# general form. pandas_dataframe.column.loc[true_false_index_same_size_as_dataframe]
positive_pledges = kickstarters_2017.usd_pledged_real.loc[index_of_positive_pledges]

# normalize the pledges (w/ Box-Cox)
# note that positive_pledges is of type: pandas.core.series.Series
# note that normalized_pledges is of type: numpy.ndarray
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
# this is a great python statement
# it will assign True or False to each 'row' based on the condition 
index_of_pledged = kickstarters_2017.pledged > 0

# get only positive pledges (using their indexes)
# this fun piece of code picks out all of the usd_pledged_real values ONLY IF they are positive,
# as found within the index_of_positive_pledges
# general form. pandas_dataframe.column.loc[true_false_index_same_size_as_dataframe]
positive_pledged = kickstarters_2017.usd_pledged_real.loc[index_of_pledged]

# normalize the pledges (w/ Box-Cox)
# note that positive_pledges is of type: pandas.core.series.Series
# note that normalized_pledges is of type: numpy.ndarray
normalized_pledged = stats.boxcox(positive_pledged)[0]

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(positive_pledged, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_pledged, ax=ax[1])
ax[1].set_title("Normalized data")

# I'm not sure what this 'pledged' data is telling us 
# compared with the one before 'usd_pledged_real'
# however, after normalizing, it looks the same