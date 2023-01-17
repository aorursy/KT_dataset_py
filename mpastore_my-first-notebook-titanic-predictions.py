# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# load training data into dataframe

train_data = pd.read_csv('../input/train.csv')

train_data.head()
# Basic details about training data (to start orienting myself in dataset)

train_data.describe()
#So from here I can glean...

# 1) 891 samples in this dataset

# 2) "Survived" is boolean, presumed 0 for no, 1 for yes

# 3) There are only 714 age values for 891 passengers, meaning 177 passengers are missing ages. Investigate...

# 4) Average age was 29.6 with standard deviation of 14.5 years



# Check datatypes

train_data.info()
# How many people survived? 



#Remove any non-survivors



survivors = train_data.loc[train_data['Survived'] == 1]

len(survivors)
# Create subset with only Survived and Age

survivors_age = train_data[['Survived', 'Age']]

survivors_age.head()
survivors_age = survivors['Age']

survivors_age.plot.hist()
# See scatterplot of men vs. women & survival

survivors.head()
survivors_gender = survivors[['Sex', 'Age']]

survivors_gender.plot.hist()
# Correlation between age and fare paid?

age_fare = train_data[['Age', 'Fare']]

age_fare.info()
# Replace NaN with 100 (makes it easier to see outliers)

age_fare.fillna(100, inplace=True)

age_fare.head(20)
# Scatterplot of age vs. fare

age_fare.plot.scatter(x='Age', y='Fare')
#What's age breakout of those with cheap fares?

unknownage_fares = age_fare.loc[age_fare['Age'] == 100]

unknownage_fares['Fare'].plot.hist()
# From the histogram above, I'd deduce that many of the free fares are those with unknown ages (or in this model, age = 100)

# Potential stowaways who didn't record age, and also "paid" free fare? Maybe.