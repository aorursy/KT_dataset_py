# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns
df = pd.read_csv("../input/hotel-booking-demand/hotel_bookings.csv")
df.describe()
df.head(5)
#Making a copy of our data

data = df
#Getting an overview of the number of unique values present in each of the columns

data.nunique()
# Names of the two hotels

data.hotel.unique()
null_check = data.isnull().sum()

cols_with_missing_values = null_check[null_check != 0]

cols_with_missing_values
# deleting rows with null values for 'children' and 'country'

data.dropna(subset = ['children', 'country'], inplace = True)
'''to preserve important information for 'agent' and 'company' columns, 

  we'll replace the null values with 0 since both columns contain numerical data.'''

data.agent.fillna(0, inplace = True)

data.company.fillna(0, inplace = True)
#verifying whether we have handled the null values or not

null_check = data.isnull().sum()

cols_with_missing_values = null_check[null_check != 0]

cols_with_missing_values
import seaborn as sns

import matplotlib.pyplot as plt
#Getting the frequency of booking of each country into a Data Frame

country_dist = data.groupby('country').count()['hotel']

country_dist = pd.DataFrame(country_dist)



#The Data Frame has columns "Country" and "No. of Bookings"

country_dist['Country'] = country_dist.index

country_dist = country_dist.rename(columns = {'hotel': 'No. of Bookings'})



#Sorting the DataFrame in descending order and getting only those countries which have bookings more than 1000.

country_dist = country_dist.sort_values(by = 'No. of Bookings', ascending = False)

popular_country_dist = country_dist[country_dist['No. of Bookings'] > 1000]
popular_country_dist.head(5)
sns.barplot(x = popular_country_dist['Country'][1:], y = popular_country_dist['No. of Bookings'][1:])
#Getting the monthly frequency of cancelled and successful bookings 

monthly_dist = data[data.is_canceled == 0].groupby('arrival_date_month').count()['hotel']

monthly_cancelled_dist = data[data.is_canceled == 1].groupby('arrival_date_month').count()['hotel']
'''Storing the data into two dataframes and concatenating both of them to get a single dataframe with columns "No. of Bookings"

"Month" and "is_canceled"'''

monthly_dist = pd.DataFrame(monthly_dist)

monthly_cancelled_dist = pd.DataFrame(monthly_cancelled_dist)



monthly_dist = monthly_dist.rename(columns = {"hotel" : "No. of Bookings"})

monthly_cancelled_dist = monthly_cancelled_dist.rename(columns = {"hotel" : "No. of Bookings"})



monthly_dist['is_canceled'] = 'No'

monthly_cancelled_dist['is_canceled'] = 'Yes'



monthly_dist["Month"] = monthly_dist.index

monthly_cancelled_dist["Month"] = monthly_cancelled_dist.index



monthly_freq = pd.concat([monthly_dist, monthly_cancelled_dist])
months_in_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
ax = sns.factorplot("Month", "No. of Bookings", col="is_canceled", data=monthly_freq, kind="bar", order = months_in_order)

print(type(ax))

#ax.set(title = 'Monthly Distribution of Bookings acc. to the status of booking.')

ax.set_xticklabels(rotation=45)

ax1,ax2 = ax.axes[0]

ax1.axhline(7000, ls = '--', linewidth = 2)

ax2.axhline(5000, ls = '--', linewidth = 2)

plt.show()

data.head(2)
lead_time = data['lead_time']

lead_time = pd.DataFrame(sorted(lead_time, reverse = True), columns = ['Lead'])

sns.distplot(lead_time)
a4_dims = (21, 6)

fig, ax = plt.subplots(1,3,figsize=a4_dims)

sns.distplot(lead_time[lead_time['Lead'] < 100], ax = ax[0])

sns.distplot(lead_time[(lead_time['Lead'] > 100) & (lead_time['Lead'] < 365)], ax = ax[1])

sns.distplot(lead_time[lead_time['Lead'] > 365], ax = ax[2])
data.head(2)