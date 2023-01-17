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

sns.distplot(kickstarters_2017.usd_goal_real, ax=ax[0])

ax[0].set_title("Original Data")

sns.distplot(scaled_data, ax=ax[1])

ax[1].set_title("Scaled data")
# Your turn! 



# We just scaled the "usd_goal_real" column. What about the "goal" column?

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
# Your turn! 

# We looked as the usd_pledged_real column. What about the "pledged" column? Does it have the same info?

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

from sklearn.preprocessing import minmax_scale





# Any results you write to the current directory are saved as output.
weather_data=pd.read_csv("../input/weatherww2/Summary of Weather.csv")
weather_data.head()
weather_data.describe()
weather_data.describe(include=['O'])
#There are 119040 rows and 31 columns.

print (weather_data.shape)

print ('--'*30)



#Lets see what percentage of each column has null values

#It means count number of nulls in every column and divide by total num of rows.



print (weather_data.isnull().sum()/weather_data.shape[0] * 100)
cols = [col for col in weather_data.columns if (weather_data[col].isnull().sum()/weather_data.shape[0] * 100 < 70)]

weather_data_trimmed = weather_data[cols]



#STA is more of station code, lets drop it for the moment

weather_data_trimmed = weather_data_trimmed.drop(['STA'], axis=1)



print ('Legitimate columns after dropping null columns: %s' % weather_data_trimmed.shape[1])
weather_data_trimmed.isnull().sum()
weather_data_trimmed.sample(5)
#Check dtypes and look for conversion if needed

weather_data_trimmed.dtypes



#Looks like some columns needs to be converted to numeric field



weather_data_trimmed['Snowfall'] = pd.to_numeric(weather_data_trimmed['Snowfall'], errors='coerce')

weather_data_trimmed['SNF'] = pd.to_numeric(weather_data_trimmed['SNF'], errors='coerce')

weather_data_trimmed['PRCP'] = pd.to_numeric(weather_data_trimmed['PRCP'], errors='coerce')

weather_data_trimmed['Precip'] = pd.to_numeric(weather_data_trimmed['Precip'], errors='coerce')



weather_data_trimmed['Date'] = pd.to_datetime(weather_data_trimmed['Date'])
#Fill remaining null values. FOr the moment lts perform ffill



weather_data_trimmed.fillna(method='ffill', inplace=True)

weather_data_trimmed.fillna(method='bfill', inplace=True)



weather_data_trimmed.isnull().sum()

#Well no more NaN and null values to worry about
print (weather_data_trimmed.dtypes)

print ('--'*30)

weather_data_trimmed.sample(3)
#weather_data_trimmed_scaled = minmax_scale(weather_data_trimmed.iloc[:, 1:])



weather_data_trimmed['Precip_scaled'] = minmax_scale(weather_data_trimmed['Precip'])

weather_data_trimmed['MeanTemp_scaled'] = minmax_scale(weather_data_trimmed['MeanTemp'])

weather_data_trimmed['YR_scaled'] = minmax_scale(weather_data_trimmed['YR'])

weather_data_trimmed['Snowfall_scaled'] = minmax_scale(weather_data_trimmed['Snowfall'])

weather_data_trimmed['MAX_scaled'] = minmax_scale(weather_data_trimmed['MAX'])

weather_data_trimmed['MIN_scaled'] = minmax_scale(weather_data_trimmed['MIN'])
#Plot couple of columns to see how the data is scaled



fig, ax = plt.subplots(4, 2, figsize=(15, 15))



sns.distplot(weather_data_trimmed['Precip'], ax=ax[0][0])

sns.distplot(weather_data_trimmed['Precip_scaled'], ax=ax[0][1])



sns.distplot(weather_data_trimmed['MeanTemp'], ax=ax[1][0])

sns.distplot(weather_data_trimmed['MeanTemp_scaled'], ax=ax[1][1])



sns.distplot(weather_data_trimmed['Snowfall'], ax=ax[2][0])

sns.distplot(weather_data_trimmed['Snowfall_scaled'], ax=ax[2][1])



sns.distplot(weather_data_trimmed['MAX'], ax=ax[3][0])

sns.distplot(weather_data_trimmed['MAX_scaled'], ax=ax[3][1])

from scipy.stats import boxcox



Precip_norm = boxcox(weather_data_trimmed['Precip_scaled'].loc[weather_data_trimmed['Precip_scaled'] > 0])

MeanTemp_norm = boxcox(weather_data_trimmed['MeanTemp_scaled'].loc[weather_data_trimmed['MeanTemp_scaled'] > 0])

YR_norm = boxcox(weather_data_trimmed['YR_scaled'].loc[weather_data_trimmed['YR_scaled'] > 0])

Snowfall_norm = boxcox(weather_data_trimmed['Snowfall_scaled'].loc[weather_data_trimmed['Snowfall_scaled'] > 0])

MAX_norm = boxcox(weather_data_trimmed['MAX_scaled'].loc[weather_data_trimmed['MAX_scaled'] > 0])

MIN_norm = boxcox(weather_data_trimmed['MIN_scaled'].loc[weather_data_trimmed['MIN_scaled'] > 0])
fig, ax = plt.subplots(4, 2, figsize=(15, 15))



sns.distplot(weather_data_trimmed['Precip_scaled'], ax=ax[0][0])

sns.distplot(Precip_norm[0], ax=ax[0][1])



sns.distplot(weather_data_trimmed['MeanTemp_scaled'], ax=ax[1][0])

sns.distplot(MeanTemp_norm[0], ax=ax[1][1])



sns.distplot(weather_data_trimmed['Snowfall_scaled'], ax=ax[2][0])

sns.distplot(Snowfall_norm[0], ax=ax[2][1])



sns.distplot(weather_data_trimmed['MAX_scaled'], ax=ax[3][0])

sns.distplot(MAX_norm[0], ax=ax[3][1])