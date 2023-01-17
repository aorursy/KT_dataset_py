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

#https://www.dataquest.io/blog/python-vs-r/
import random
filename = "../input/train.csv"
samples = 100000

# Read some samples - TAKES A LOT TO COMPUTE
#n = sum(1 for line in open(filename)) - 1 #excludes header
#print(n)
n = 37670294
#skip = sorted(random.sample(range(1,n+1),n-samples)) #the 0-indexed header will not be included in the skip list

#train_data = pd.read_csv(filename, parse_dates=['date_time', 'srch_ci', 'srch_co'], skiprows=skip)
#train_data.info()

train_data = pd.read_csv(filename, parse_dates=['date_time', 'srch_ci', 'srch_co'], nrows=samples)
train_data.info()

print('---------------')
print('We use %.1f%% of a data' % (samples/n))
train_data.head(5)
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(y='hotel_continent', data=train_data)
sns.plt.title('Continent destination')
plt.show() #get rid of <matplotlib.text.Text at 0x7f ...
f, ax = plt.subplots(figsize=(15, 25))
sns.countplot(y='hotel_country', data=train_data)
plt.title('Search of hotels per country')
plt.show()
sns.pairplot(train_data[['hotel_country', 'user_location_country']], size=6)
plt.show()
train_data.loc[:,['is_mobile', 'is_package', 'orig_destination_distance', 'srch_adults_cnt', 
                  'srch_children_cnt', 'srch_rm_cnt', 'is_booking', 'cnt']].describe()
#https://www.dataquest.io/blog/python-data-science/
train_data.groupby('hotel_continent').count().sort_values(by='hotel_country')
corr = train_data.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 15))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, vmax=.3, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
plt.show()
#Divide data into training and test
from sklearn.model_selection import train_test_split
train, test, y_train, y_test = train_test_split(
    train_data[['hotel_continent', 'user_location_country', 'hotel_country']], 
    train_data['hotel_cluster'], 
    test_size=0.33, 
    random_state=13)
#display samle data with label
train.head().join(y_train.head())
#Fit a model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, min_samples_leaf=10)
model.fit(train, y_train)
#mean squared error
from sklearn.metrics import mean_squared_error
import math

predictions = model.predict(test)
mean_squared_error(predictions, y_test)