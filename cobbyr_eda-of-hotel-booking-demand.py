#setup imports 



import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import seaborn as sns

import numpy as np

%matplotlib inline 

import os



import warnings

warnings.filterwarnings('ignore')
#there are two different hotels being Resort and City hotel 
#read CSV

hotel = pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')

hotel.head()
#search for null values in the dataframe 

hotel.isnull().sum()
#this opens up the notebook so that you can view all the columns with no limits. 

pd.set_option('display.max_columns', None)



#replace missing values and dropping columns.  

Nan = {'country': 'Unknown', 'children': 0, 'agent': 0}

hotel = hotel.fillna(Nan)



#changing values from unknown to SC in meal column

hotel['meal'].replace('Undefined', 'SC', inplace=True)
#dropping columns that have too many null values: Company column has 94% null values so it is dropped for analysis) 

hotel = hotel.drop(['company'], 1)

hotel.head()
hotel.isnull().sum()
#describing categorical data 

hotel.describe(include=["O"])
#viewing the measures of tendency 

hotel.describe()
#finding the anomly 

hotel[hotel['adr'] == (-6.38)] 
#find the anomly's and replace 

hotel = hotel.drop([48515])



hotel = hotel.drop([14969])
#there are some a few records that have zero average daily rate(ADR) and are a no-show

#These type of records will be excluded from the dataset since they dont provide any insight. 

hotel.loc[(hotel["adr"]==0) & (hotel["reservation_status"]=="No-Show")]
#checking for data quality 

hotel.is_canceled.value_counts()
#removing data that has values of 0 for ADR and status of no-show for reseveration status

df = hotel.loc[(hotel["adr"]!=0) & (hotel["reservation_status"]!="No-Show")]

df
#view the data to see what is left after preprocessing

df.shape
df.describe(include=["O"])
df = df.reset_index(drop=True)

df
#creating Dataframe that excludes the large number of cancelations 

df2 = df[df.is_canceled == 0]

df2
df.groupby('arrival_date_month')['arrival_date_year'].unique()
#getting the data

#create split DF for city and resort hotel 

resortdf = df[df['hotel'] == 'Resort Hotel']

resortdf



citydf = df[df['hotel'] == 'City Hotel']

citydf



#get total counts for each month for each city 

mresortdata = resortdf.groupby('arrival_date_month')['hotel'].count()

mcitydata =  citydf.groupby('arrival_date_month')['hotel'].count()



mresortdf = pd.DataFrame({'hotel': 'Resort Hotel','month': list(mresortdata.index),

                          'guests':list(mresortdata.values)})

mcitydf = pd.DataFrame({'hotel': 'City Hotel','month': list(mcitydata.index),

                          'guests':list(mcitydata.values)})

#concat to combine the two hotel data for easy viewing 

monthlydf = pd.concat([mresortdf, mcitydf], ignore_index = True)



#order the months for appropriate ordered viewing 

months = ["January", "February", "March", "April", "May", "June", 

"July", "August", "September", "October", "November", "December"]



monthlydf['month'] = pd.Categorical(monthlydf['month'], categories = months, ordered= True)



#normalizing the data 



monthlydf.loc[(monthlydf["month"] == "July") | (monthlydf["month"] == "August"),

                    "guests"] /= 3

monthlydf.loc[(monthlydf["month"] != "July") | (monthlydf["month"] != "August"),

                    "guests"] /= 2

#graphing

plt.figure(figsize=(10, 8))

sns.set(style = 'darkgrid')

sns.lineplot(x='month', y= 'guests', hue = 'hotel' , data = monthlydf, sort = False)

plt.xticks( rotation= 50)

plt.legend(loc='upper right')

plt.title('Number of Guests Per Month')

plt.show()
#distribution of average daily rates

print("Skewness: %.2f" % df['adr'].skew())

print("Kurtosis: %.2f" % df['adr'].kurt())

plt.figure(figsize=(10, 8))

sns.distplot(df['adr'])

sns.set(style = 'darkgrid')

plt.title('Average Daily Rate Distribution')

plt.xlabel('ADR (EUR€)')

plt.ion()

plt.show()
plt.figure(figsize=(10, 8))

sns.set(style="darkgrid")

htc = sns.catplot(x="customer_type", y="adr", hue="hotel", data=df,

height=6, kind="bar", palette="muted")

htc.despine(left=True)

htc.set_ylabels("ADR (EUR€)")

htc.set_xlabels("Cusotmer Type")

plt.title('Customer Type Prices')

plt.ion()

plt.show()
plt.figure(figsize=(10, 8))

sns.jointplot(x="lead_time", y="adr", data=df, s = 10)

plt.show()
months = ["January", "February", "March", "April", "May", "June", 

"July", "August", "September", "October", "November", "December"]



month_revenue = pd.Categorical(df['arrival_date_month'], categories = months, ordered= True)

plt.figure(figsize=(10, 8))

sns.lineplot(x=month_revenue, y= df.adr, hue = 'hotel' , data = df, sort = False)

plt.xticks( rotation= 50)

plt.ylabel('ADR (EUR€)')

plt.title('Average Daily Rate Per Month')

plt.show()
#barplot 

sns.set(style="darkgrid", palette="pastel")

plt.figure(figsize=(10, 8))

htc = sns.catplot(x="total_of_special_requests", y="adr", hue="hotel", data=df,

height=6, kind="bar", palette="muted")

htc.despine(left=True)

htc.set_ylabels("ADR (EUR€)")

htc.set_xlabels("Number of Special Requests")

plt.show()
#reserveed room type and average daily rate 

sns.set(style="darkgrid")

plt.figure(figsize=(10, 8))

htc = sns.catplot(x="reserved_room_type", y="adr", hue="hotel", data=df,

                height=6, kind="bar", palette="muted")

htc.despine(left=True)

htc.set_ylabels("ADR (EUR€)")

htc.set_xlabels("Reserved Room Type")

plt.title('Average Daily Rate of Room Types')

plt.show()
#leadtime cancelations using df

plt.figure(figsize=(10, 8))

sns.lineplot(x=month_revenue, y= 'is_canceled', hue = 'hotel' , data = df, sort = False)

plt.xticks( rotation= 50)

plt.ylabel('Cancelations')

plt.title('Cancelations Per Month')

plt.show()
#finding values of top visitors for both resorts

topc_resort = df[df['hotel']=="Resort Hotel"]["country"].value_counts().head(10)

topc_city = df[df['hotel']=="City Hotel"]["country"].value_counts().head(10)

topc = pd.concat([topc_city,topc_resort],axis=1)

topc.columns = ["city","resort"]

topc
new_topc = topc.rename_axis('country').reset_index()

#create df for resort values 

new_topr = new_topc.drop('city', 1)

new_topr.sort_values(['resort'], ascending=False, inplace= True)

new_topr.reset_index(drop=True)
plt.figure(2, figsize=(20,15))

the_grid = gridspec.GridSpec(2, 2)



plt.subplot(the_grid[0, 1],  title='Top Visitors from City Hotel')

sns.barplot(x='country',y='city', data=new_topc, palette='Spectral')

plt.ylabel('Number of Visitors')



plt.subplot(the_grid[0, 0], title='Top Visitors from Resort Hotel',)

sns.barplot(x='country',y='resort', data=new_topr, palette='Spectral')

plt.ylabel('Number of Visitors')



plt.suptitle('Top Countries Visiting ', fontsize=16)

plt.show()
pie = df['country'].value_counts().head(10)



labels = ['PRT','GBR','FRA','ESP','DEU','ITA','IRL','BEL','BRA','NLD']



fig, ax = plt.subplots()

ax.pie(pie, labels = labels ,autopct='%1.1f%%', shadow=True)

plt.title('Top 10 Vistitors')

plt.figure(figsize=(10, 8))

plt.show()
#duration of stay  

df2['totalday'] = df2['stays_in_weekend_nights'] + df2['stays_in_week_nights']

df2.head()
print("Skewness: %.2f" % df2['totalday'].skew())

print("Kurtosis: %.2f" % df2['totalday'].kurt())

plt.figure(figsize=(10, 8))

sns.distplot(df2['totalday'])

plt.xlabel('Total Days Spent')

plt.show()
#creating revenue column 

#average daily rate times days spent = room revenue 



df2['Revenue'] = df2.adr * df2.totalday

df2.groupby("hotel")["Revenue"].describe()
#show figure 

plt.figure(figsize=(10, 8))

f, ax = plt.subplots(figsize=(6.5, 6.5))

sns.scatterplot(x='totalday', y= 'adr', hue = 'hotel',palette="ch:r=-.2,d=.3_r", 

                data = df2 , ax = ax, sizes=(1,10))

plt.xlabel('Total Days Spent')

plt.ylabel('ADR (EUR€)')

plt.show()
plt.figure(figsize=(10, 8))

sns.lineplot(x=df['assigned_room_type'], y= df2.Revenue, hue = 'hotel' , data = df2)

plt.xlabel('Assigned Room Type')

plt.show()