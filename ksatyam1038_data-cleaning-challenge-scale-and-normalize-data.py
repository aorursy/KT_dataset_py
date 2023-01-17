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
Seattle_pet_licenses = pd.read_csv("../input/seattle-pet-licenses/seattle_pet_licenses.csv")
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
sns.distplot(kickstarters_2017.usd_goal_real, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")
kickstarters_2017.head(5)
Seattle_pet_licenses.head(5)
# Your turn! 

# Using goal column 
goal = kickstarters_2017.usd_pledged_real
#scaling goal coloum from 0 to 1 
scaled_data = minmax_scaling(goal, columns= [0])

                             
#plot the orignal and scaled data together to campare                             
fig, ax=plt.subplots(1,2)
sns.distplot(kickstarters_2017.goal, ax=ax[0])
ax[0].set_title("Orignal Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled Data")

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

#Normalization using pledged coloumns

#let's first get index of all positive pledged
index_of_positive_pledges=kickstarters_2017.pledged>0
#only positive pledged
positive_pledges= kickstarters_2017.pledged.loc[index_of_positive_pledges]


#now normalize the pledged
normalized_pledges = stats.boxcox(positive_pledges)[0]
#plot both togther to campare
fig, ax=plt.subplots(1,2)


sns.distplot(positive_pledges ,ax=ax[1])
ax[0].set_title("Orignal data")
sns.distplot(normalized_pledges, ax= ax[1])

ax[1].set_title("Normalized data")
