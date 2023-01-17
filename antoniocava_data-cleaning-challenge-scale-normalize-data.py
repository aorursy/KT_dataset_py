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
kickstarters_2017.head(5) # look at the data
kickstarters_2017.shape # how many rows/columns?
# generate 1000 data points randomly drawn from an exponential distribution
original_data = np.random.exponential(size = 1000)

# min-max scale the data between 0 and 1
scaled_data = minmax_scaling(original_data, columns = [0]) # Note that 'columns=[0]' is needed! otherwise get error
# NOTE: this might be a bit misleading (ok, maybe just to me ;-) )  since usually one may think that scaling 
# refers to scale/modify the distribution itself, while here we are scaling the 'range' only

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0]) # plot distribution/histogram
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")
original_data.shape
#scaled_data.shape
# normalize the exponential data with boxcox
normalized_data = stats.boxcox(original_data)
# NOTE: Again, wath many people might be used to is to 'normalize distributions' in the sense that you get a scaled-to-max-1 distribution 
# or an integrated-equal-to-1 distribution area, while here 'normalization' refers  to the transformation from a given ditribution 
# to a 'normal/gaussian' distribution. 

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

# We can still plot original and scaled data, however this does not make too much sense since the currencies are different amongst the sample.

# select the usd_goal_real column
goal = kickstarters_2017.goal

# scale the goals from 0 to 1
scaled_goal = minmax_scaling(goal, columns = [0])

# plot the original & scaled data together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(kickstarters_2017.goal, ax=ax[0])
ax[0].set_title("Original Data (goal)")
sns.distplot(scaled_goal, ax=ax[1])
ax[1].set_title("Scaled data (goal)")

# get the index of all positive pledges (Box-Cox only takes postive values)
index_of_positive_pledges = (kickstarters_2017.usd_pledged_real > 0) # & (kickstarters_2017.usd_pledged_real < 10000) # put upper limit

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

# get the index of all positive pledges (Box-Cox only takes postive values)
index_of_positive_pledges = (kickstarters_2017.pledged > 0) # & (kickstarters_2017.usd_pledged_real < 10000) # put upper limit

# get only positive pledges (using their indexes)
positive_pledges = kickstarters_2017.pledged.loc[index_of_positive_pledges]

# normalize the pledges (w/ Box-Cox)
normalized_pledges = stats.boxcox(positive_pledges)[0]

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(positive_pledges, ax=ax[0], color='g') # try changing colors
ax[0].set_title("Original Data")
sns.distplot(normalized_pledges, ax=ax[1],color='y')
ax[1].set_title("Normalized data")

# We looked at the usd_pledged_real column. What about the "pledged" column? Does it have the same info?


