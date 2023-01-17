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
pwd
# generate 1000 data points randomly drawn from an exponential distribution
original_data = np.random.exponential(size = 1000)
#original_data = np.random.randn(1000)

# mix-max scale the data between 0 and 1
scaled_data = minmax_scaling(original_data, columns = [0])

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")
original_databis=original_data.reshape(-1,1)
original_databis.shape
from sklearn.preprocessing import MinMaxScaler,StandardScaler
Norm=MinMaxScaler()
norm_scaled_data=Norm.fit_transform(original_databis)

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(original_databis, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(norm_scaled_data, ax=ax[1])
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
kickstarters_2017.usd_goal_real.shape, kickstarters_2017.columns
# Your turn! 

# We just scaled the "usd_goal_real" column. What about the "goal" column?
usd_goal = kickstarters_2017.goal

# scale the goals from 0 to 1
scaled_data = minmax_scaling(usd_goal, columns = [0])

# plot the original & scaled data together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(kickstarters_2017.goal, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")
index_of_positive_pledges = kickstarters_2017.usd_pledged_real > 0
index_of_positive_pledges.shape, kickstarters_2017.shape
#kickstarters_2017.usd_pledged_real.loc[index_of_positive_pledges]
positive_pledges = kickstarters_2017.usd_pledged_real.loc[index_of_positive_pledges]
positive_pledges.shape
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
index_of_positive_pledges = kickstarters_2017.pledged > 0
index_of_positive_pledges.shape
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
