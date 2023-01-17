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
import pandas as pd

building_metadata = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv")

sample_submission = pd.read_csv("../input/ashrae-energy-prediction/sample_submission.csv")

test = pd.read_csv("../input/ashrae-energy-prediction/test.csv")

train = pd.read_csv("../input/ashrae-energy-prediction/train.csv")

weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv")

weather_train = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv")
train.describe()
# Merging data sets. The pd.merge() function recognizes that each DataFrame has an "employee" column, and automatically joins using this column as a key. 

# Merging training data & building meta data



df = pd.merge(train, building_metadata)



# listing the columns

list(df.columns) 
# Merging df with weather data

dff = pd.merge(df, weather_train)



# listing the columns

list(dff.columns)
# checking data types 



print(dff.dtypes)
# Looking at distributions of the columns using data.hist feature & defining the size of the plot using matplotlib



import matplotlib.pyplot as plt



fig = plt.figure(figsize = (10,10))

ax = fig.gca()

dff.hist(ax=ax)

plt.show()
# Meter reading has a peak - replotting histogram



dff.hist(column='meter_reading', bins = 5)



# Looks like there are outliers
dff.describe()
# Removing outliers in meter reading

# For your dataframe column, you could get quantile with:

q = dff["meter_reading"].quantile(0.95)



#and then filter with:



dfout = dff[dff["meter_reading"] < q]

dfout.hist(column='meter_reading')
# Precep depth also seems to have outliers looking at the histogram - cleaning that with same steps as above



# Removing outliers in precipitation depth

# For your dataframe column, you could get quantile with:

r = dfout["precip_depth_1_hr"].quantile(0.99)



#and then filter with:



dfouts = dfout[dfout["precip_depth_1_hr"] < r]

dfouts.hist(column='precip_depth_1_hr')

# Merging test data sets as well using the same steps



# Merging data sets. The pd.merge() function recognizes that each DataFrame has an "employee" column, and automatically joins using this column as a key. 

# Merging training data & building meta data



#dft = pd.merge(test, building_metadata)



# Merging df with weather data

#dfft = pd.merge(dft, weather_test)



# listing the columns

#list(dfft.columns)