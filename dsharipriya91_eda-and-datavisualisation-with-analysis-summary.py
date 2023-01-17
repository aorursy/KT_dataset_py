#check handholding example for categorical variable
#next EDA for a numerical value
import pandas as pd
import matplotlib as plt
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
df = pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')
df.columns
df.shape
df.sample(5)
df.describe()
df['hotel'].value_counts(normalize=True)
df['is_canceled'].value_counts(normalize=True)
#Check for any extreme outliers. If you want to plot a boxplot, create a new variable without this value
df['adr'].sort_values(ascending=False)
#Single extreme outlier at index 48515 that can distort the visibility of a boxplot
df_adr_no_outlier = df[df.index!=48515]
sns.boxplot(x=df_adr_no_outlier['adr'])
df_adr_no_outlier['adr'].median()
#New variable creted for calculating total nights stayed
df['total_nights'] = df['stays_in_weekend_nights']+df['stays_in_week_nights']

#Considering only those who didnt cancel
df_stayed = df[df['is_canceled']==0]

plt.figure(figsize=(15,8))
# filtering the data for a clearer plot
filter = df_stayed['total_nights']<10

sns.countplot(df_stayed[filter]['total_nights'], hue=df['hotel'])
#added hue='hotel' to check how the average nights vary
df['country'].value_counts(normalize=True).head(10)
df['company'].value_counts().sort_values(ascending=False).head(10)

sns.countplot(x=df['distribution_channel'], hue=df['hotel'])
df['distribution_channel'].value_counts(normalize=True)
plt.figure(figsize=(10,6))
sns.countplot(y= 'market_segment', data = df)
df['is_repeated_guest'].value_counts(normalize=True)
plt.figure(figsize=(20,10))
sns.countplot(df['lead_time'])
df['lead_time'].describe()
pd.crosstab(df["hotel"], df['is_canceled'], normalize=True).plot(kind='bar', stacked=True)
sns.boxplot(data=df, x='is_canceled', y='lead_time')
ct=pd.crosstab(df['previous_cancellations'],df['is_canceled'], normalize='index')
ct.plot.bar(stacked=True)
ct=pd.crosstab([df['previous_cancellations'],df['distribution_channel']],df['is_canceled'], normalize='index')
ct.plot.bar(stacked=True)
plt.figure(figsize=(20,10))
sns.scatterplot(data=df, x='days_in_waiting_list', y='booking_changes', hue='is_canceled' )
#booking changes has also been added to the analysis
plt.figure(figsize=(15,8))
sns.boxplot(y=df_adr_no_outlier['distribution_channel'], x=df_adr_no_outlier['adr'], hue=df['hotel'])
#The variation is split hotelwise also gain more information
plt.figure(figsize=(15,8))
sns.boxplot(data=df_adr_no_outlier, x='market_segment', y='adr')
plt.figure(figsize=(15,8))
sns.lineplot(data=df_adr_no_outlier, x='arrival_date_month', y='adr', hue='hotel', sort=False)
#data without the one extreme adr outlier is used here
plt.figure(figsize=(15,8))
sns.countplot(data=df_adr_no_outlier, x='arrival_date_month', hue='hotel')
#data without the one extreme adr outlier is used here
plt.figure(figsize=(15,8))
sns.lineplot(data=df, x='arrival_date_month', y='is_canceled', hue='hotel', sort=False)
pd.crosstab(df['is_canceled'],df['total_of_special_requests'], normalize='columns')
filter_roomtype = df['reserved_room_type'] != df['assigned_room_type']
df[filter_roomtype]['is_canceled'].value_counts(normalize=True)
pd.crosstab(df['deposit_type'],df['is_canceled'], normalize='index').plot(kind='bar', stacked=True)
