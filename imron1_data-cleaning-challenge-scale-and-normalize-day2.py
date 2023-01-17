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
sns.distplot(kickstarters_2017.usd_goal_real, ax=ax[0],color="green")
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1],color="cyan")
ax[1].set_title("Scaled data")
# Your turn! 
kickstarters_2017.sample(10)
kickstarters_2017x=kickstarters_2017.copy()
#print ("")
#print ("Normalized the data")
#print(kickstarters_2017x.shape)
#print ("")
##print ("knowing the detail of the data statistic attribute")
#print (kickstarters_2017x.describe())
#print ("")
#print ("knowing how the data looks like at a glance")
#print (kickstarters_2017x.sample(10))
##print ("")
#print ("knowing the data types for each column")
#print (kickstarters_2017.dtypes)
# We just scaled the "usd_goal_real" column. What about the "goal" column?

# select the real column
goalx = kickstarters_2017x.goal

# scale the goals from 0 to 1
scaled_data = minmax_scaling(goalx, columns = [0])

# plot the original & scaled data together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(kickstarters_2017.goal, ax=ax[0],color="green")
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1],color="cyan")
ax[1].set_title("Scaled data")

# get the index of all positive pledges (Box-Cox only takes postive values)
index_of_positive_pledges = kickstarters_2017.usd_pledged_real > 0

# get only positive pledges (using their indexes)
positive_pledges = kickstarters_2017.usd_pledged_real.loc[index_of_positive_pledges]

# normalize the pledges (w/ Box-Cox)
normalized_pledges = stats.boxcox(positive_pledges)[0]

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(positive_pledges, ax=ax[0],color="green")
ax[0].set_title("Original Data")
sns.distplot(normalized_pledges, ax=ax[1],color="black")
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
sns.distplot(positive_pledges, ax=ax[0],color="green")
ax[0].set_title("Original Data")
sns.distplot(normalized_pledges, ax=ax[1],color="black")
ax[1].set_title("Normalized data")