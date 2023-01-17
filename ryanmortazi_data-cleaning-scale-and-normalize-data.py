# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# for min_max scaling

from sklearn.preprocessing import MinMaxScaler



# plotting modules

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
scaler=MinMaxScaler()

# Let's create a data for purpose of explaining.

# generate 1000 data points randomly drawn from an exponential distribution

original_data = np.random.exponential(size = 1000)



# mix-max scale the data between 0 and 1

scaled_data = scaler.fit_transform(original_data.reshape(-1,1))



# plot both together to compare

fig, ax=plt.subplots(1,2)

sns.distplot(original_data, ax=ax[0])

ax[0].set_title("Original Data")

sns.distplot(scaled_data, ax=ax[1])

ax[1].set_title("Scaled data")
# for Box-Cox Transformation

from scipy import stats

# normalize the exponential data with boxcox

normalized_data = stats.boxcox(original_data)



# plot both together to compare

fig, ax=plt.subplots(1,2)

sns.distplot(original_data, ax=ax[0])

ax[0].set_title("Original Data")

sns.distplot(normalized_data[0], ax=ax[1])

ax[1].set_title("Normalized data")
data = pd.read_csv("../input/new-data/data.csv")

data.head()
# select the usd_goal_real column

usd_goal = data.usd_goal_real

# scale the goals from 0 to 1

scaled_data_2 = scaler.fit_transform(original_data.reshape(-1,1))



# plot the original & scaled data together to compare

fig, ax=plt.subplots(1,2)

sns.distplot(usd_goal, ax=ax[0])

ax[0].set_title("Original Data")

sns.distplot(scaled_data_2, ax=ax[1])

ax[1].set_title("Scaled data")

# get the index of all positive pledges (Box-Cox only takes postive values)

index_of_positive_pledges = data.usd_pledged_real > 0



# get only positive pledges (using their indexes)

positive_pledges = data.usd_pledged_real.loc[index_of_positive_pledges]



# normalize the pledges (w/ Box-Cox)

normalized_pledges = stats.boxcox(positive_pledges)[0]



# plot both together to compare

fig, ax=plt.subplots(1,2)

sns.distplot(positive_pledges, ax=ax[0])

ax[0].set_title("Original Data")

sns.distplot(normalized_pledges, ax=ax[1])

ax[1].set_title("Normalized data")