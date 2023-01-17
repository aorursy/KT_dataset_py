import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy import stats # for box-cox tracnsformantion

from mlxtend.preprocessing import minmax_scaling # for minmax scaling

import seaborn as sns

import matplotlib.pyplot as plt



np.random.seed(0) # for reproducibility
df = pd.read_csv('/kaggle/input/kickstarter-projects/ks-projects-201801.csv')
# generate 1000 data points randomly drawn from an exponential distribution

original_data = np.random.exponential(size = 1000)



# min-max scale the data between 0-1

scaled_data = minmax_scaling(original_data, columns = [0])



# plot both together to compare

fig, ax = plt.subplots(1,2)

sns.distplot(original_data, ax = ax[0])

ax[0].set_title("Original Data")

sns.distplot(scaled_data, ax = ax[1])

ax[1].set_title("Scaled data")
# normalize the exponential data with boxcox

normalized_data = stats.boxcox(original_data)



# plot both together to compare

fig, ax = plt.subplots(1, 2)

sns.distplot(original_data, ax = ax[0])

ax[0].set_title("Original data")

sns.distplot(normalized_data[0], ax = ax[1])

ax[1].set_title("Normalized data")
# select the usd_goal_real column

usd_goal = df.usd_goal_real



# scale the goals from 0 to 1

scaled_data = minmax_scaling(usd_goal, columns = [0])



# plot the original and scaled data to compare

fig, ax = plt.subplots(1,2)

sns.distplot(df.usd_goal_real, ax = ax[0])

ax[0].set_title("Original data")

sns.distplot(scaled_data, ax = ax[1])

ax[1].set_title("Scaled data")
df.columns
# select the goal column

goal = df.goal



# scale the goals from 0 to 1

scaled_data = minmax_scaling(goal, columns = [0])



# plot the original and scaled data to compare

fig, ax = plt.subplots(1,2)

sns.distplot(df.goal, ax = ax[0])

ax[0].set_title("Original data")

sns.distplot(scaled_data, ax = ax[1])

ax[1].set_title("Scaled data")
# get the index of all positive pledges (Box-Cox only takes postive values)

index_of_positive_pledges = df.usd_pledged_real > 0



# get only positive pledges (using their indexes)

positive_pledges = df.usd_pledged_real.loc[index_of_positive_pledges]



# normalize the pledges (w/ Box-Cox)

normalized_pledges = stats.boxcox(positive_pledges)[0]



# plot both together to compare

fig, ax=plt.subplots(1,2)

sns.distplot(positive_pledges, ax=ax[0])

ax[0].set_title("Original Data")

sns.distplot(normalized_pledges, ax=ax[1])

ax[1].set_title("Normalized data")
# get the index of all positive pledges (Box-Cox only takes postive values)

index_of_positive_pledges = df.pledged > 0



# get only positive pledges (using their indexes)

positive_pledges = df.pledged.loc[index_of_positive_pledges]



# normalize the pledges (w/ Box-Cox)

normalized_pledges = stats.boxcox(positive_pledges)[0]



# plot both together to compare

fig, ax=plt.subplots(1,2)

sns.distplot(positive_pledges, ax=ax[0])

ax[0].set_title("Original Data")

sns.distplot(normalized_pledges, ax=ax[1])

ax[1].set_title("Normalized data")