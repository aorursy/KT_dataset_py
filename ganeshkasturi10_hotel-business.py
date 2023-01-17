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
# Ignore warnings

import warnings  

warnings.filterwarnings('ignore')
import pandas as pd

data = pd.read_csv("/kaggle/input/hotel-booking-demand/hotel_bookings.csv")
data.head()
data.tail()
data.info()
data.hotel.unique()
data.deposit_type.unique()
data.customer_type.unique()
data.reservation_status.unique()
data.shape
data.describe()
missing_value_presentage =round(data.isnull().sum()*100/len(data),2).reset_index()

missing_value_presentage.columns = ['column_name','missing_value_presentage']

missing_value_presentage = missing_value_presentage.sort_values('missing_value_presentage',ascending =False)

missing_value_presentage
#Removing columns having missing value % is >=50

data.drop('company',axis=1,inplace=True)
#Remove null values 

data.dropna(inplace=True)
missing_value_presentage =round(data.isnull().sum()*100/len(data),2).reset_index()

missing_value_presentage.columns = ['column_name','missing_value_presentage']

missing_value_presentage = missing_value_presentage.sort_values('missing_value_presentage',ascending =False)

missing_value_presentage
# Hotel Type :

import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize = (7,5))

ax = sns.countplot(x ='hotel',data = data)

plt.title('Hotel Type')

plt.xlabel('Hotel')

plt.ylabel('Total Booking')

for p in ax.patches:

    ax.annotate((p.get_height()),(p.get_x()+0.4 , p.get_height()+100)) 
# Sometimes customers cancel their reservation due to various reasons. Let's see how many of them have canceled.

plt.figure(figsize=(8,8))

ax = sns.countplot(x="is_canceled", data = data, palette="RdYlGn")

plt.title('Is Canceled?')

plt.xlabel('Is Canceled?')

plt.ylabel('Total Count')

for p in ax.patches:

    ax.annotate((p.get_height()),(p.get_x()+0.4 , p.get_height()+100)) 

# One-Third of the users have canceled their reservation.
plt.figure(figsize=(14,6))

ax = sns.countplot(x ='hotel',data =data,hue='is_canceled',palette='pastel')

plt.show()

for p in ax.patches:

    ax.annotate((p.get_height()),(p.get_x()+0.4 , p.get_height()+100)) 
cross_1 =pd.crosstab(data.hotel,data.is_canceled)

cross_1
data['cancel_accordingyear'] = data['arrival_date_year']+ data['is_canceled'] 

ax = sns.countplot(x="cancel_accordingyear", data = data, palette="RdYlGn")

plt.title('Most Year of Cancel')

plt.xlabel('Is_Canceled')

plt.ylabel('Total Count')

for p in ax.patches:

    ax.annotate((p.get_height()),(p.get_x()+0.4 , p.get_height()+100)) 
month_sorted = ['January','February','March','April','May','June','July','August','September','October','November','December']

plt.figure(figsize=(14,6))

sns.countplot(data['arrival_date_month'], palette='pastel', order = month_sorted)

plt.xticks(rotation = 90)

plt.show()
perc_monthly_canc = pd.DataFrame(data[data['is_canceled'] == 1]['arrival_date_month'].value_counts() * 100 / data['arrival_date_month'].value_counts())

perc_monthly_canc.reset_index()

plt.figure(figsize=(14,6))

sns.barplot(x=perc_monthly_canc.index,y='arrival_date_month',data=perc_monthly_canc, order=month_sorted, palette='pastel')

plt.xticks(rotation = 90)

plt.ylabel('% cancellation per month')

plt.show()
plt.figure(figsize=(8,8))

ax = sns.countplot(x="customer_type", data = data, palette="RdYlGn")

plt.title('Customes Type')

plt.xlabel('Types of Customers')

plt.ylabel('Total Count')

for p in ax.patches:

    ax.annotate((p.get_height()),(p.get_x()+0.4 , p.get_height()+100)) 
data[['adults','children','babies']] = data[['adults','children','babies']].fillna(0).astype(int)

data['total_guests'] = data['adults']+ data['children']+ data['babies']

plt.figure(figsize=(12,8))

ax = sns.countplot(x="total_guests", data = data,palette = 'twilight_shifted')

plt.title('Number of Guests')

plt.xlabel('total_guests')

plt.ylabel('Count')

for p in ax.patches:

    ax.annotate((p.get_height()),(p.get_x()+0.1 , p.get_height()+100))
plt.figure(figsize=(12,8))

ax = sns.countplot(x="market_segment", data = data,palette = 'magma',order = data['market_segment'].value_counts().index)

plt.title('Market Segment')

plt.xlabel('market_segment')

plt.ylabel('Count')

for p in ax.patches:

    ax.annotate((p.get_height()),(p.get_x()+0.2 , p.get_height()+100))
plt.figure(figsize=(12,8))

ax = sns.countplot(x="distribution_channel", data =data,palette = 'viridis',order = data['distribution_channel'].value_counts().index)

plt.title('Distribution Channel')

plt.xlabel('distribution_channel')

plt.ylabel('Count')

for p in ax.patches:

    ax.annotate((p.get_height()),(p.get_x()+0.3 , p.get_height()+100)) 
plt.figure(figsize=(12,8))

ax = sns.countplot(x="is_repeated_guest", data = data, palette="RdYlGn")

plt.title('Is Repeated Guest?')

plt.xlabel('is_repeated_guest')

plt.ylabel('Total Count')

for p in ax.patches:

    ax.annotate((p.get_height()),(p.get_x()+0.4 , p.get_height()+100))
plt.figure(figsize=(12,8))

ax = sns.countplot(x="required_car_parking_spaces", data = data, palette="jet_r",order = data['required_car_parking_spaces'].value_counts().index)

plt.title('Total Car Parking Spaces Required')

plt.xlabel('required_car_parking_spaces')

plt.ylabel('Total Count')

for p in ax.patches:

    ax.annotate((p.get_height()),(p.get_x()+0.35 , p.get_height()+100))
plt.figure(figsize=(12,8))

ax = sns.countplot(x="deposit_type", data = data, palette="jet_r",order = data['deposit_type'].value_counts().index)

plt.title('Deposit Type')

plt.xlabel('deposit_type')

plt.ylabel('Total Count')

for p in ax.patches:

    ax.annotate((p.get_height()),(p.get_x()+0.35 , p.get_height()+100)) 

cross_2 =pd.crosstab(data.deposit_type,data.is_canceled)

cross_2
data['total_nights_stayed'] = data['stays_in_weekend_nights'] + data['stays_in_week_nights']

plt.figure(figsize=(20,8))

ax = sns.countplot(x="total_nights_stayed", data = data, palette="tab10")

plt.title('Total Nights Stayed')

plt.xlabel('total_nights_stayed')

plt.ylabel('Total Count')

for p in ax.patches:

    ax.annotate((p.get_height()),(p.get_x()-0.1 , p.get_height()+100)) 
plt.figure(figsize=(20,8))

ax = sns.countplot(x="reservation_status", data = data, palette="tab20")

plt.title('Reservation Status')

plt.xlabel('reservation_status')

plt.ylabel('Total Count')

for p in ax.patches:

    ax.annotate((p.get_height()),(p.get_x()+0.35 , p.get_height()+100))
cross_3 =pd.crosstab(data.reservation_status,data.is_canceled)

cross_3
fig,axes = plt.subplots(1,1,figsize=(10,7))

sns.heatmap(data.corr())

plt.show()