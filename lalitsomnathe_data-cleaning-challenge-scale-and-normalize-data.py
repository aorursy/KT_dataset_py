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

from IPython.display import display
display(kickstarters_2017.head(2))
kickstarters_2017.info()
kickstarters_2017.describe(include='all')
# generate 1000 data points randomly drawn from an exponential distribution
original_data = np.random.exponential(size = 1000)
#print(original_data)
# mix-max scale the data between 0 and 1
scaled_data = minmax_scaling(original_data, columns = [0])
#print(scaled_data)
# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")
#OUT OF CURIOSITY
from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaler = StandardScaler()
#sklearn.MinMaxScaler works with atleast 2D data. original_data is series.
temp=pd.DataFrame(np.random.exponential(size = 1000).reshape(500,2))
temp1=scaler.fit_transform(temp)
plt.figure()
plt.subplot(121)
sns.distplot(temp[1])

plt.subplot(122)
sns.distplot(temp1[1])
# here it is normally distributed .. Strange!
# normalize the exponential data with boxcox
normalized_data = stats.boxcox(original_data)

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_data[0], ax=ax[1])
ax[1].set_title("Normalized data")
#OUT OF CURIOSITY
from sklearn.preprocessing import Normalizer,normalize
#norm_sk= normalize(original_data[:,np.newaxis], axis=0).ravel()
norm_sk=normalize(temp)
fig,ax =plt.subplots(1,2)
sns.distplot(temp[0], ax=ax[0])

sns.distplot(norm_sk[0], ax=ax[1])
# sklearn normalizer need atleast 2d data , original_data is 
#THis is exactl equal to sklearn scaling technique.. Strange!
# select the usd_goal_real column
usd_goal = kickstarters_2017.usd_goal_real

# scale the goals from 0 to 1
scaled_data = minmax_scaling(usd_goal, columns = [0])

# plot the original & scaled data together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(usd_goal, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")
# Your turn! 

# We just scaled the "usd_goal_real" column. What about the "goal" column?

goals=kickstarters_2017.goal
scaled_goals=minmax_scaling(goals,columns=[0])
plt.figure()
plt.subplot(121)
sns.distplot(goals)
plt.subplot(122)
sns.distplot(scaled_goals)
# get the index of all positive pledges (Box-Cox only takes postive values)
index_of_positive_pledges = kickstarters_2017.usd_pledged_real > 0
#print(index_of_positive_pledges.sample(2))
# get only positive pledges (using their indexes)
positive_pledges = kickstarters_2017.usd_pledged_real.loc[index_of_positive_pledges]
#positive_pledges.sample(2)
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
pled=kickstarters_2017[kickstarters_2017['pledged']>0].pledged
pled_norm=stats.boxcox(pled)[0]

fig,ax=plt.subplots(1,2)
sns.distplot(pled, ax=ax[0])
sns.distplot(pled_norm, ax=ax[1])
