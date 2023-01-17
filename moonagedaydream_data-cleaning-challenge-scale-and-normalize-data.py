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

# min-max scale the data between 0 and 1
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

# play around a bit at first
kickstarters_2017[:3]
kickstarters_2017[kickstarters_2017.usd_pledged_real > 1.0e+07]
kickstarters_2017.usd_pledged_real.describe()
maxi = kickstarters_2017.usd_pledged_real.max()
kickstarters_2017[kickstarters_2017.usd_pledged_real==maxi]
print (usd_goal.mean())
print (usd_goal.median())
print (usd_goal.mode())
# select the usd_goal_real column
usd_goal = kickstarters_2017.usd_goal_real

# scale the goals from 0 to 1
scaled_data = minmax_scaling(usd_goal, columns = [0])

# plot the original & scaled data together to compare
fig, ax=plt.subplots(1,2, figsize=(8,4))
sns.distplot(kickstarters_2017.usd_goal_real, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")
# Your turn! 

# We just scaled the "usd_goal_real" column. What about the "goal" column?
goal_scaled = minmax_scaling(kickstarters_2017.goal, columns=[0] )

fig, ax = plt.subplots(1,2)
sns.distplot(kickstarters_2017.goal, ax=ax[0])
ax[0].set_title("original: goal")
sns.distplot(goal_scaled, ax=ax[1])
ax[1].set_title("scaled [0-1] ")
kickstarters_2017.head(2)
# get the index of all positive pledges (Box-Cox only takes postive values)
index_of_positive_pledges = kickstarters_2017.goal > 0

# get only positive pledges (using their indexes)
positive_pledges = kickstarters_2017.goal.loc[index_of_positive_pledges]

# normalize the pledges (w/ Box-Cox)
normalized_pledges = stats.boxcox(positive_pledges)[0]

# plot both together to compare
fig, ax=plt.subplots(1,2, figsize=(10,4))
sns.distplot(positive_pledges, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_pledges, ax=ax[1])
ax[1].set_title("Normalized data")
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

kickstarters_2017.pledged.describe()
kickstarters_2017[kickstarters_2017.pledged==0].pledged.count()
# should 
kickstarters_2017.pledged.count()
kickstarters_2017.pledged.loc
# Your turn! 
# We looked as the usd_pledged_real column. What about the "pledged" column? Does it have the same info?

# only positive values; remove the zero values 
index_pos_pledged = [kickstarters_2017.pledged > 0]

positive_pledged = kickstarters_2017.pledged.loc[kickstarters_2017.pledged > 0]

# normalize the pledges (w/ Box-Cox)
normalized_pledges = stats.boxcox(positive_pledged)[0]

# plot the pledged column
fig, ax = plt.subplots(1,2, figsize=(8,4))
sns.distplot(kickstarters_2017.pledged, ax=ax[0])
ax[0].set_title("Original - pledged")
sns.distplot(normalized_pledges, ax=ax[1])
ax[1].set_title("Normalized-pledged")