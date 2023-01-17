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
import numpy as np

import pandas as pd



from datetime import datetime



import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline





!pip install jovian --upgrade --quiet

import jovian
project='hotel_booking_demand_course_project'

# jovian.commit(project=project)
df = pd.read_csv("/kaggle/input/hotel-booking-demand/hotel_bookings.csv")
df
df.sample(10)
df.shape
df.describe()
df.info()
# Checking for null values

df.isnull().sum()
df.drop('agent', axis=1, inplace=True)

df.drop('company', axis=1, inplace=True)
print(df.hotel.value_counts())
# Only non cancelled bookings included 

guests_per_country = df[df.is_canceled == 0].groupby('country').hotel.count().sort_values(ascending = False)

guests_per_country = guests_per_country.reset_index()

guests_per_country.rename(columns = {'hotel' : 'bookings'}, inplace = True)





# The top 10 countries with highest number of bookings

guests_per_country_top_10 = guests_per_country[:10].copy()



# Others countries bookings combined

new_row = pd.DataFrame(data = { 'country' : ['Others'],

                                'bookings' : [guests_per_country['bookings'][10:].sum()]

                                })



guests_per_country = pd.concat([guests_per_country_top_10, new_row])





#Pie chart Plot

fig = plt.figure(figsize =(15, 15)) 

plt.title("Percentage of guests from countries")

plt.legend(guests_per_country.index, loc="best")

plt.pie(guests_per_country.bookings , autopct='%1.1f%%', labels=guests_per_country.country , explode =(0.1,0,0,0,0,0,0,0,0,0,0));
print('{0:.2f}% of all bookings are cancelled.'.format(df.is_canceled.value_counts()[1]*100/df.is_canceled.count()))
df['arrival_date'] = df.apply(lambda x:datetime.strptime("{0} {1} {2}".format(x['arrival_date_year'],x['arrival_date_month'], x['arrival_date_day_of_month']), "%Y %B %d"),axis=1)

df[["arrival_date", "arrival_date_day_of_month","arrival_date_month","arrival_date_year"]].sample(5) 
# Get total monthly bookings

bookings_per_month = df.groupby(df['arrival_date'].dt.strftime('%B-%Y')).count().hotel  # convert arrival date to month-year format to count total no of bookings in a month

bookings_per_month = bookings_per_month.reset_index()



bookings_per_month['month_year'] = pd.to_datetime(bookings_per_month.arrival_date, format='%B-%Y')  # Convert month-year to datetime and create new column to sort values

bookings_per_month = bookings_per_month.sort_values('month_year')  



print(bookings_per_month.head())





# Plot monthly bookings 



plt.figure(figsize=(20, 8))

plt.title("Total Monthly bookings done")

plt.xlabel("Date")

plt.ylabel("Number of bookings")

plt.plot(bookings_per_month.arrival_date , bookings_per_month.hotel);

plt.xticks(rotation=45);


bookings_per_month = df.groupby(['hotel',  df['arrival_date'].dt.strftime('%B-%Y')]).count().lead_time

bookings_per_month = bookings_per_month.reset_index()

bookings_per_month.rename(columns = {'lead_time':'bookings'} , inplace = True)

bookings_per_month

bookings_per_month['month_year'] = pd.to_datetime(bookings_per_month.arrival_date, format='%B-%Y')  # Convert to datetime to sort values

bookings_per_month = bookings_per_month.sort_values('month_year')  

# print(bookings_per_month.head())



# bookings_per_month = bookings_per_month.unstack()

bookings_per_month_rh = bookings_per_month[bookings_per_month.hotel == 'Resort Hotel']

bookings_per_month_ch = bookings_per_month[bookings_per_month.hotel == 'City Hotel']





plt.figure(figsize=(20, 8))

plt.title("Total Monthly bookings of both the hotels compared")

plt.xlabel("Months")

plt.ylabel("Total no of bookings")

plt.plot(bookings_per_month_rh.arrival_date , bookings_per_month_rh.bookings);

plt.plot(bookings_per_month_ch.arrival_date , bookings_per_month_ch.bookings);

plt.legend(["Resort Hotel","City Hotel"])

plt.xticks(rotation=45);

                            

# Bookings done including cancelled bookings made by different types of customers.



# df.customer_type.unique()

customer_bookings = df.groupby(['customer_type','is_canceled']).count().hotel.reset_index()

customer_bookings.rename(columns = {'hotel':'bookings'} , inplace = True)

print(customer_bookings)



# Plotting The graph

plt.figure(figsize = (10,6))



ax = sns.barplot('customer_type', 'bookings', hue='is_canceled', data=customer_bookings );



plt.title("Number of booking made / cancelled by different types of customers")

plt.xlabel("Customer Type")

plt.ylabel("Number of bookings ")



leg_handles = ax.get_legend_handles_labels()[0]

ax.legend(leg_handles, ['Not Cancelled', 'Booking Cancelled'], title='Cancelled');
# Bookings done including cancelled bookings made by different types of customers.



# df.customer_type.unique()

customer_bookings = df.groupby(['customer_type','hotel']).count().lead_time.reset_index()

customer_bookings.rename(columns = {'lead_time':'bookings'} , inplace = True)

print(customer_bookings)



# Plotting The graph

plt.figure(figsize = (10,6))



sns.barplot('customer_type', 'bookings', hue='hotel', data=customer_bookings );



plt.title("Hotel preference by customer type")

plt.xlabel("Customer Type")

plt.ylabel("Number of bookings");

# df.lead_time

# Plot histogram for 90 days or 3 months of lead time



sns.set_style("whitegrid")

plt.figure(figsize = (20,6))

plt.hist(df.lead_time, bins=np.arange(0, 90, 1));





plt.xlabel("Number of days between Booking date and Arrival Date")

plt.ylabel("Number of Bookings");
meal_preference = df.meal.value_counts()

meal_preference.index.name = "meal_type"

meal_preference = meal_preference.reset_index()

meal_preference



plt.figure(figsize = (15,10))

plt.pie(meal_preference.meal, autopct='%1.1f%%', labels=meal_preference.meal_type);



plt.title("Meal Preference Pie chart");

deposit_type_by_hotel  = df.groupby(["hotel","deposit_type"]).lead_time.count()

deposit_type_by_hotel = deposit_type_by_hotel.reset_index()

deposit_type_by_hotel.rename(columns = {"lead_time":"bookings"}, inplace = True)

deposit_type_by_hotel



plt.figure(figsize = (10,6))



sns.barplot('deposit_type', 'bookings', hue='hotel', data=deposit_type_by_hotel );

plt.title("Bookings by Deposit type")

plt.xlabel("Deposit type")

plt.ylabel("Number of bookings");
print('{0:.2f}% of all bookings are cancelled at City Hotel.'.format(df[df.hotel == "City Hotel"].is_canceled.sum() * 100 / df[df.hotel == "City Hotel"].is_canceled.count()))

print('{0:.2f}% of all bookings are cancelled at Resort Hotel.'.format(df[df.hotel == "Resort Hotel"].is_canceled.sum() * 100 / df[df.hotel == "Resort Hotel"].is_canceled.count()))
# Checking for corelation between columns using heat map



# sns.set_style("whitegrid")

# plt.figure(figsize = (20,20))

# sns.heatmap(df.corr(), annot=True, cmap='Blues'); 
project='hotel_booking_demand_course_project'



# jovian.commit(project=project)