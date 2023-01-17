# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Import training data - review shape and columns

import pandas as pd

train = pd.read_csv('/kaggle/input/learn-together/train.csv')

train.shape

train.columns
# Reverse the one hot encoding for Wilderness Area and Soil Type to create one column each for Wilderness Areas and Soil Types which contain all all possible category values

train_1 = train.copy()

train_1['Wild_Area'] = train_1.iloc[:,11:15].idxmax(axis=1)

train_1['Soil_Type'] = train_1.iloc[:,16:55].idxmax(axis=1)



# Drop the one hot encoded columns to a get a data frame only with numeric and the categorical columns

col_to_drop = np.arange(11,55)

col_to_drop

train_1.drop(train_1.columns[col_to_drop], axis = 1, inplace = True)

train_1.columns

train_1.head()
# check the distribution of Wild Area

train_1.Wild_Area.value_counts()
# check the distribution of Forest Cover Types

train_1.Cover_Type.value_counts()
# check the distribution of Soil Types

train_1.Soil_Type.value_counts()
# check the avail_soil types which have counts less than 100. 

less_100 = train_1.Soil_Type.value_counts() < 100

less_100_soil = train_1.Soil_Type.value_counts()[less_100].index

less_100_soil

# list of non zero soil types available in training data

avail_soil = train_1.Soil_Type.value_counts().index

avail_soil



# Get all possible Soil Types available as features in the training data as columns

all_soil = train.columns[15:55]

all_soil

# Check the missing soil types from training data by comparing all_soil and avail_soil

miss_soil = np.setdiff1d(all_soil,avail_soil)

miss_soil
# Create list of Soil type columns to be removed - missing soil types + less than 50 counts soil types

remove_soil = list(miss_soil) + list(less_100_soil)

remove_soil
# Create box plots for all numeric variables - Elevation, Aspect, Slope, Horizontal Distance to Hydrology

#import seaborn as sns

#import matplotlib.pyplot as plt

#sns.set(rc={'figure.figsize':(15,15)})

#fig,ax = plt.subplots(5,2)

#sns.boxplot(x = 'Cover_Type', y = 'Elevation', data = train_1, ax = ax[0,0])

#sns.boxplot(x = 'Cover_Type', y = 'Aspect', data = train_1, ax = ax[0,1])

#sns.boxplot(x = 'Cover_Type', y = 'Slope', data = train_1,ax = ax[1,0])

#sns.boxplot(x = 'Cover_Type', y = 'Horizontal_Distance_To_Hydrology', data = train_1,ax = ax[1,1])

#sns.boxplot(x = 'Cover_Type', y = 'Vertical_Distance_To_Hydrology', data = train_1,ax = ax[2,0])

#sns.boxplot(x = 'Cover_Type', y = 'Horizontal_Distance_To_Roadways', data = train_1,ax = ax[2,1])

#sns.boxplot(x = 'Cover_Type', y = 'Horizontal_Distance_To_Fire_Points', data = train_1,ax = ax[3,0])

#sns.boxplot(x = 'Cover_Type', y = 'Hillshade_9am', data = train_1,ax = ax[3,1])

#sns.boxplot(x = 'Cover_Type', y = 'Hillshade_Noon', data = train_1,ax = ax[4,0])

#sns.boxplot(x = 'Cover_Type', y = 'Hillshade_3pm', data = train_1,ax = ax[4,1])

#plt.close(4,1)

#plt.close(3)

#plt.close(4)

#plt.close(5)

#plt.yticks(rotation = 90)

# Plot box plots of all numeric variables by Cover Type

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(rc={'figure.figsize':(10,5)})

for column in train_1.columns[1:10]:

    sns.boxplot(x = 'Cover_Type', y = column, data = train_1)

    plt.show()
# Plot Hillshade variables

import seaborn as sns

#train_1['Cover_Type'] = train_1['Cover_Type'].astype('object')

sns.scatterplot(x='Hillshade_9am', y = 'Hillshade_3pm', data = train_1, hue = 'Cover_Type', palette=sns.color_palette("Set1", 7))

plt.show()

sns.scatterplot(x='Hillshade_9am', y = 'Hillshade_Noon', data = train_1, hue = 'Cover_Type', palette=sns.color_palette("Set1", 7))

plt.show()

sns.scatterplot(x='Hillshade_Noon', y = 'Hillshade_3pm', data = train_1, hue = 'Cover_Type', palette=sns.color_palette("Set1", 7))

plt.show()

#train_1.plot(x='Hillshade_3pm', y = 'Hillshade_Noon', kind = 'scatter', c = 'Cover_Type')
# construct scatter plots to check for 2 variable combinations that can separate the classes

sns.pairplot(train_1, hue = 'Cover_Type')
#sns.scatterplot(x='Aspect', y = 'Horizontal_Distance_To_Fire_Points', data = train_1, hue = 'Cover_Type', palette=sns.color_palette("Set1", 7))

#sns.scatterplot(x='Hillshade_Noon', y = 'Elevation', data = train_1, hue = 'Cover_Type', palette=sns.color_palette("Set1", 7))