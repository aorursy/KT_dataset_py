import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime

pd.set_option('display.max_columns',None)

df = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')

df.head()
df.shape # Number of rows and columns
df.describe()
percentage_missing_values = round(df.isnull().sum()*100/len(df),2).reset_index()

percentage_missing_values.columns = ['column_name','percentage_missing_values']

percentage_missing_values = percentage_missing_values.sort_values('percentage_missing_values',ascending = False)

percentage_missing_values
plt.figure(figsize=(12,8))

ax = sns.countplot(x="hotel", data = df)

plt.title('Hotel Type')

plt.xlabel('Hotel')

plt.ylabel('Total Bookings')

for p in ax.patches:

    ax.annotate((p.get_height()),(p.get_x()+0.4 , p.get_height()+100)) 
plt.figure(figsize=(12,8))

ax = sns.countplot(x="is_canceled", data = df, palette="RdYlGn")

plt.title('Is Canceled?')

plt.xlabel('Is Canceled?')

plt.ylabel('Total Count')

for p in ax.patches:

    ax.annotate((p.get_height()),(p.get_x()+0.4 , p.get_height()+100)) 
def month_converter(month):

    months = ['January', 'February', 'March', 'April', 'May', 'June','July', 'August', 'September', 'October', 'November', 'December']

    return months.index(month) + 1

df['arrival_month'] = df['arrival_date_month'].apply(month_converter)

df['arrival_year_month'] = df['arrival_date_year'].astype(str) + " _ " + df['arrival_month'].astype(str)



plt.figure(figsize=(24,8))

ax = sns.countplot(x="arrival_year_month", data = df, palette="CMRmap_r")

plt.title('Arrival Year_Month')

plt.xlabel('arrival_year_month')

plt.ylabel('Total Count')

for p in ax.patches:

    ax.annotate((p.get_height()),(p.get_x()+0.2 , p.get_height()+50)) 
df['Arrrival Date'] = df.apply(lambda row: datetime.strptime(f"{int(row.arrival_date_year)}-{int(row.arrival_month)}-{int(row.arrival_date_day_of_month)}", '%Y-%m-%d'), axis=1)

df['arrival_day_of_week'] = df['Arrrival Date'].dt.day_name()

weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

df['arrival_day_of_week'] = pd.Categorical(df['arrival_day_of_week'],categories = weekdays)

arrivals = pd.pivot_table(df,columns = 'arrival_day_of_week',index = 'arrival_month',values = 'reservation_status',aggfunc = 'count')

fig, ax = plt.subplots(figsize = (16,11))

ax = sns.heatmap(arrivals ,annot=True, fmt="d",cmap = 'rocket_r')
df[['adults','children','babies']] = df[['adults','children','babies']].fillna(0).astype(int)

df['total_guests'] = df['adults']+ df['children']+ df['babies']

plt.figure(figsize=(12,8))

ax = sns.countplot(x="total_guests", data = df,palette = 'twilight_shifted')

plt.title('Number of Guests')

plt.xlabel('total_guests')

plt.ylabel('Count')

for p in ax.patches:

    ax.annotate((p.get_height()),(p.get_x()+0.1 , p.get_height()+100)) 
plt.figure(figsize=(20,8))

df_country = df['country'].value_counts().nlargest(25).astype(int)

ax = sns.barplot(df_country.index, df_country.values)

plt.title('Country')

plt.xlabel('Country')

plt.ylabel('Count')

for p in ax.patches:

    ax.annotate((p.get_height()),(p.get_x(), p.get_height()+100)) 
plt.figure(figsize=(12,8))

ax = sns.countplot(x="market_segment", data = df,palette = 'magma',order = df['market_segment'].value_counts().index)

plt.title('Market Segment')

plt.xlabel('market_segment')

plt.ylabel('Count')

for p in ax.patches:

    ax.annotate((p.get_height()),(p.get_x()+0.2 , p.get_height()+100)) 
plt.figure(figsize=(12,8))

ax = sns.countplot(x="distribution_channel", data = df,palette = 'viridis',order = df['distribution_channel'].value_counts().index)

plt.title('Distribution Channel')

plt.xlabel('distribution_channel')

plt.ylabel('Count')

for p in ax.patches:

    ax.annotate((p.get_height()),(p.get_x()+0.3 , p.get_height()+100)) 
plt.figure(figsize=(12,8))

ax = sns.countplot(x="is_repeated_guest", data = df, palette="RdYlGn")

plt.title('Is Repeated Guest?')

plt.xlabel('is_repeated_guest')

plt.ylabel('Total Count')

for p in ax.patches:

    ax.annotate((p.get_height()),(p.get_x()+0.4 , p.get_height()+100)) 
plt.figure(figsize=(12,8))

ax = sns.countplot(x="customer_type", data = df, palette="nipy_spectral",order = df['customer_type'].value_counts().index)

plt.title('Customer Type')

plt.xlabel('customer_type')

plt.ylabel('Total Count')

for p in ax.patches:

    ax.annotate((p.get_height()),(p.get_x()+0.3 , p.get_height()+100)) 
plt.figure(figsize=(12,8))

ax = sns.countplot(x="required_car_parking_spaces", data = df, palette="jet_r",order = df['required_car_parking_spaces'].value_counts().index)

plt.title('Total Car Parking Spaces Required')

plt.xlabel('required_car_parking_spaces')

plt.ylabel('Total Count')

for p in ax.patches:

    ax.annotate((p.get_height()),(p.get_x()+0.35 , p.get_height()+100)) 
plt.figure(figsize=(12,8))

ax = sns.countplot(x="deposit_type", data = df, palette="jet_r",order = df['deposit_type'].value_counts().index)

plt.title('Deposit Type')

plt.xlabel('deposit_type')

plt.ylabel('Total Count')

for p in ax.patches:

    ax.annotate((p.get_height()),(p.get_x()+0.35 , p.get_height()+100)) 
df['total_nights_stayed'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']

plt.figure(figsize=(20,8))

ax = sns.countplot(x="total_nights_stayed", data = df, palette="tab10")

plt.title('Total Nights Stayed')

plt.xlabel('total_nights_stayed')

plt.ylabel('Total Count')

for p in ax.patches:

    ax.annotate((p.get_height()),(p.get_x()-0.1 , p.get_height()+100)) 
plt.figure(figsize=(20,8))

ax = sns.countplot(x="reservation_status", data = df, palette="tab20")

plt.title('Reservation Status')

plt.xlabel('reservation_status')

plt.ylabel('Total Count')

for p in ax.patches:

    ax.annotate((p.get_height()),(p.get_x()+0.35 , p.get_height()+100)) 