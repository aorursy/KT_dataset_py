# modules we'll use
import pandas as pd
import numpy as np

# for Box-Cox Transformation
from scipy import stats

# for min_max scaling
from mlxtend.preprocessing import minmax_scaling
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import normalize

# plotting modules
import seaborn as sns
import matplotlib.pyplot as plt

# read in all our data
kickstarters_2017 = pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv")

# set seed for reproducibility
np.random.seed(0)
#kickstarters_2017.sample(5)
kickstarters_2017.info()
#kickstarters_2017.isnull().sum()
# generate 1000 data points randomly drawn from an exponential distribution
original_data = np.random.exponential(size = 1000)
print ('Min and Max from original dataset %f %f' % (np.amin(original_data), np.amax(original_data)))

# mix-max scale the data between 0 and 1
scaled_data = minmax_scaling(original_data, columns = [0])
print ('Min and Max from scaled dataset %f %f' % (np.amin(scaled_data), np.amax(scaled_data)))
print (scaled_data[0:5])

sklearn_func_scaled_data = minmax_scale(original_data)
print ('Min and Max from sklearn scale function dataset %f %f' % (np.amin(sklearn_func_scaled_data), np.amax(sklearn_func_scaled_data)))
print (sklearn_func_scaled_data[0:5])

original_copy = original_data.copy()
sklearn_minmax_scaler = MinMaxScaler()
sklearn_class_scaled_data = sklearn_minmax_scaler.fit_transform(original_copy.reshape(-1, 1))
print ('Min and Max from sklearn MinMaxScaler dataset %f %f' % (np.amin(sklearn_class_scaled_data), np.amax(sklearn_class_scaled_data)))
print (sklearn_class_scaled_data[0:5])

print ('--'*30)

fig, ax=plt.subplots(2,2, figsize=(6, 10))

ax[0][0].set_title('Original data')
sns.distplot(original_data, ax=ax[0][0])

ax[0][1].set_title('mlxtend.preprocessing.minmax_scaling')
sns.distplot(scaled_data, ax=ax[0][1])

ax[1][0].set_title('minmax_scale')
sns.distplot(sklearn_func_scaled_data, ax=ax[1][0])

ax[1][1].set_title('MinMaxScaler')
sns.distplot(sklearn_class_scaled_data, ax=ax[1][1])

'''
# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")
'''
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
#goal = kickstarters_2017.goal
goal = kickstarters_2017[['goal']]
scaled_goal = minmax_scale(goal)

print ('Min Max of original goal %d %d' % (np.amin(goal), np.amax(goal)))
print ('Min Max of scaled goal %d %d' % (np.amin(scaled_goal), np.amax(scaled_goal)))

fig, ax = plt.subplots(1,2, figsize=(5, 5))
sns.distplot(goal, ax=ax[0])
ax[0].set_title('Original goal data')
sns.distplot(scaled_goal, ax=ax[1])
ax[1].set_title('Scaled goal data')
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
#Here I will try to use sklearn.preprocessing.normalize and see whats the distribution is going to be

pos_pledged = kickstarters_2017[['pledged']]
norm_pledged = normalize(pos_pledged)

fig, ax = plt.subplots(1, 2, figsize=(6, 6))
sns.distplot(pos_pledged, ax=ax[0])
ax[0].set_title('pos_pledged')
sns.distplot(norm_pledged, ax=ax[1])
ax[1].set_title('sklearn.normalize')
# Your turn! 
# We looked as the usd_pledged_real column. What about the "pledged" column? Does it have the same info?

print ('Does pledged column has any null values %d ' % kickstarters_2017['pledged'].isnull().any())
print ('Does it have any -ve values %s' % ((np.amin(kickstarters_2017[['pledged']])) < 0))

index_of_positive_pledged = kickstarters_2017.pledged > 0
positive_pledged = kickstarters_2017[['pledged']].loc[index_of_positive_pledged]

normalized_pledged = stats.boxcox(positive_pledged)

fig, ax = plt.subplots(1, 2, figsize=(6, 6))
sns.distplot(positive_pledged, ax=ax[0])
ax[0].set_title('positive_pledged')
sns.distplot(normalized_pledged[0], ax=ax[1])
ax[1].set_title('normalized_pledged')