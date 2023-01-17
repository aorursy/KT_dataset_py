import numpy as np 
import pandas as pd 
pd.set_option("display.max_columns",500)

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline 
# load data:
file_path = "../input/hotel-booking-demand/hotel_bookings.csv"
data = pd.read_csv(file_path)
#take a look at the first 5 rows of the data 
data.head(5)
#let's describe the numerical data and see basic stats
data.describe()
#descibe the categorical data and see basic stats
data.describe(include="O")
#get some basic info about the data contained
data.info()
#find all the null values in the for each column 
#looks like all null values are in country, agent, and company columns
data.isnull().sum()
#For further analysis we will split the data into city and resort hotel 
city_data = data[data["hotel"]=="City Hotel"]
resort_data = data[data["hotel"]=="Resort Hotel"]
#first general pair plot to try and see relationships 
sns.distplot(city_data[city_data["adr"]<=2000]["adr"],bins=30)
plt.show()
sns.distplot(resort_data[resort_data["adr"]<=2000]["adr"],bins=30)
plt.show()
#let's see monthly adr data
monc_adr = city_data.groupby("arrival_date_month")["adr"].describe()
monc_adr = monc_adr.reindex(["January","February","March","April","May","June","July","August","September","October",\
                           "November","December"])
monc_adr
#repeat for resort data
monr_adr = resort_data.groupby("arrival_date_month")["adr"].describe()
monr_adr = monr_adr.reindex(["January","February","March","April","May","June","July","August","September","October",\
                           "November","December"])
monr_adr
mon_adr = data.groupby("arrival_date_month")["adr"].describe()
mon_adr = mon_adr.reindex(["January","February","March","April","May","June","July","August","September","October",\
                           "November","December"])
mon_adr
#higher variability with min just under $50
ax1=sns.barplot(x=monc_adr["mean"],y=monc_adr.index,palette='muted')
ax1.set_xlabel("ADR")
ax1.set_ylabel("Month")
plt.show()
#lower variability with min just over $80
ax2=sns.barplot(x=monr_adr["mean"],y=monr_adr.index,palette='muted')
ax2.set_xlabel("ADR")
ax2.set_ylabel("Month")
plt.show()
data.is_canceled.value_counts()
#create ndata variable for new data not including the cancelled bookings 
ndata = data[data.is_canceled == 0].copy()
ndata.head()
#create duration column
ndata["duration"] =  ndata['stays_in_weekend_nights'] + ndata['stays_in_week_nights']
#create revenue column
ndata["revenue"] = ndata["adr"]*ndata["duration"]
#we split the data again to look at each individually  
city_data = ndata[ndata["hotel"]=="City Hotel"]
resort_data = ndata[ndata["hotel"]=="Resort Hotel"]
#revenue data and distribution
city_data["revenue"].describe()
city_data["revenue"].sum()
resort_data["revenue"].describe()
resort_data["revenue"].sum()
#let's see monthly adr data
monc_rev = city_data.groupby("arrival_date_month").sum()["revenue"]
monc_rev = monc_rev.reindex(["January","February","March","April","May","June","July","August","September","October",\
                           "November","December"])
monc_rev
ax3=ax = monc_rev.plot.bar(rot=50)
ax3.set_xlabel("Month")
ax3.set_ylabel("Revenue")
plt.show()
#let's see monthly adr data
monr_rev = resort_data.groupby("arrival_date_month").sum()["revenue"]
monr_rev = monr_rev.reindex(["January","February","March","April","May","June","July","August","September","October",\
                           "November","December"])
monr_rev
ax4 = monr_rev.plot.bar(rot=50)
ax4.set_xlabel("Month")
ax4.set_ylabel("Revenue")
plt.show()
#revenue by channel
city_roi = city_data.groupby("distribution_channel").sum()["revenue"]
city_roi
city_roi.plot(kind="bar")
plt.show()
resort_roi = resort_data.groupby("distribution_channel").sum()["revenue"]
resort_roi
resort_roi.plot(kind="bar")
plt.show()
#distribution of special requests 
sns.countplot(data["total_of_special_requests"])
plt.show()
#most popular meal
sns.countplot(data["meal"])
plt.show()
#Most popular booking channel 
#Travel agents and tour operators are bring in most of the visitors 
sns.countplot(data["distribution_channel"])
plt.show()
#Most popular market segment 
#travel agents are most represented among our visitors as well 
sns.countplot(data["market_segment"])
plt.xticks(rotation=50)
plt.show()
top10c = city_data.country.value_counts().nlargest(10).to_frame().reset_index()
top10c.rename(columns={'index': 'Country', 'country': 'Visitors'}, inplace=True)
top10c
top10r = resort_data.country.value_counts().nlargest(10).to_frame().reset_index()
top10r.rename(columns={'index': 'Country', 'country': 'Visitors'}, inplace=True)
top10r
#Average duration of stay per month
av_dur = ndata.groupby("arrival_date_month").mean()["duration"]
av_dur = av_dur.reindex(["January","February","March","April","May","June","July","August","September","October",\
                           "November","December"])
av_dur
ax5 = av_dur.plot.bar(rot=50)
ax5.set_xlabel("Month")
ax5.set_ylabel("Duration")
plt.show()
#lead time for booking 
sns.distplot(data['lead_time'],bins=30)
plt.show()
#amount of cancellations per month
df_can = data[data["reservation_status"]=="Canceled"]
mon_can = df_can.groupby("arrival_date_month").sum()["is_canceled"]
mon_can = mon_can.reindex(["January","February","March","April","May","June","July","August","September","October",\
                           "November","December"])
mon_can
ax6 = mon_can.plot.bar(rot=50)
ax6.set_xlabel("Month")
ax6.set_ylabel("cancellations")
plt.show()
data.describe(include="O").columns
corr_data = data.drop(['hotel', 'arrival_date_month', 'meal', 'country', 'market_segment',
       'distribution_channel', 'reserved_room_type', 'assigned_room_type',
       'deposit_type', 'customer_type', 'reservation_status',
       'reservation_status_date'],axis=1)

corr_data.corr()
plt.figure(figsize=(12,6))
sns.heatmap(corr_data.corr())
corr_data.corr()["is_canceled"].sort_values(ascending=False).to_frame()
#hypothesis - city has higher lead times but also high cancellations, would reducing the lead times
#lead to reduced cancellations and high revenues
city_data["lead_time"].mean()
ocity_data = data[data["hotel"]=="City Hotel"]
oresort_data = data[data["hotel"]=="Resort Hotel"]
ocity_data["is_canceled"].sum()
city_can = ocity_data["is_canceled"].sum()/ocity_data.shape[0]
city_can
#which channel are we seeing the most cancellations 
ocity_data.groupby("distribution_channel")["is_canceled"].sum().to_frame()
oresort_data["lead_time"].mean()
oresort_data["is_canceled"].sum()
resort_can = oresort_data["is_canceled"].sum()/oresort_data.shape[0]
resort_can
#which channel are we seeing the most cancellations 
oresort_data.groupby("distribution_channel")["is_canceled"].sum().to_frame()