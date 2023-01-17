# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

import pandas_profiling
pd.reset_option('max_rows')
# to reset rc parameters to default value

plt.rcdefaults() 
df=pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')

df.head()
df.isnull().sum().plot(kind='bar')

plt.yticks(np.arange(min(df.isnull().sum()),max(df.isnull().sum()),step=5000))

plt.show()
df=df.dropna(subset=['country','agent'])
df=df.drop(columns=['company'])
df['children']=df['children'].replace(np.nan,0)

df['children'].isna().sum()
#extract importants variable

df=df[['hotel','is_canceled','arrival_date_year','arrival_date_month', 'arrival_date_week_number',

       'arrival_date_day_of_month', 'adults', 'children', 'babies','is_repeated_guest', 

       'reserved_room_type','days_in_waiting_list','required_car_parking_spaces','reservation_status']]

df.head()
df.shape
df.info()
df.describe()
df.columns.values
months = ["January", "February", "March", "April", "May", "June", 

          "July", "August", "September", "October", "November", "December"]

month_count_df=pd.DataFrame()

month_count=df['arrival_date_month'].value_counts()

month_count_df['month']=month_count.index

month_count_df['count']=month_count.values

month_count_df['month']=pd.Categorical(month_count_df['month'], categories=months, ordered=True)#to map the order so when call the order they know

month_count_df.sort_values(by=['month'],ascending=True)
for i, row in month_count_df.sort_values(by=['month'],ascending=True).iterrows():

   print( row['count'])
#plt months 

month_count_df=month_count_df.sort_values(by=['month'],ascending=True)

plt.figure(figsize=(10,5))

base_color=sns.color_palette()[0] # to set the color of the bars with the base color of pallete colors (10 colors from 0 to 9)

sns.barplot(month_count_df['month'], month_count_df['count'], alpha=0.8,color=base_color)

plt.title('plot month counts')

plt.xlabel('months', fontsize=14)

plt.ylabel('counts', fontsize=12)

plt.xticks(rotation='vertical')

plt.yticks(np.arange(0,15000,step=1000))#determine the steps(count by) 

#plt.ylim(0,13000,1000)

locs, labels = plt.xticks()

#draw text(count percent) on each bar

counter=0;

for i, row in month_count_df.iterrows():

    # get the text property for the label to get the correct count

    count = row['count']

    month = row['month']

    #print("loc {}, month {} count {}".format(row.iloc,month,count))

    pct_string = '{:0.1f}%'.format(100*count/len(df))

    # print the annotation just below the top of the bar

    plt.text(locs[counter], count - 1000, pct_string, ha = 'center', color = 'w',fontsize=12)

    counter=counter+1;

plt.show();
df.arrival_date_year.unique()
year_counts=df.arrival_date_year.value_counts()

year_counts
#another solution by using matplotlib

fig, ax = plt.subplots(figsize=(10,5), subplot_kw=dict(aspect="equal"))

wedges, texts,autotexts = ax.pie(year_counts.values, labels=year_counts.index,autopct='%1.1f%%',textprops=dict(color="w"),wedgeprops = {'width' : 0.5})

ax.legend(wedges, year_counts.index,

          title="years",

          loc="center left",

          bbox_to_anchor=(1, 0, 0.5, 1))# anchor of legend(position)

plt.setp(autotexts, size=7,weight="bold")#data inside the circle

ax.set_title("The best year in bookings",size=13,color='b')



plt.show()
def count_rows(rows):

    return len(rows)
#df.query('arrival_date_year=="2016"')['hotel'].value_counts()

by_hotel_year=df.groupby("hotel arrival_date_year".split()).apply(count_rows).unstack()

by_hotel_year
sns.heatmap(by_hotel_year)

plt.xlabel("year",fontsize=14)

plt.title("year vs hotel")
df.groupby('arrival_date_year')['hotel'].value_counts().plot(kind='bar')
hotel_cancelation_count=df.query('is_canceled==1').groupby('hotel')['is_canceled'].value_counts().unstack()

sns.heatmap(hotel_cancelation_count)
hist(df.hotel)
def func(tpl):

    if (tpl[1]==1):#cancelation

        label="Canceled {}".format(tpl[0])

        print(label)

    return label
df['intercept']=1
plt.figure(figsize = [10, 5])



# histogram on left: full data

plt.subplot(1, 2, 1)# 1 row , 2 columns , and first plot

bin_edges = np.arange(0, df['adults'].max()+2.5, 2.5)# x axis start with zero until df['skew_var'].max()+2.5 with 2.5 bin width

plt.hist(data = df, x = 'adults', bins = bin_edges)



# histogram on right: focus in on bulk of data < 5

plt.subplot(1, 2, 2)

bin_edges = np.arange(0, 5+1, 1)

plt.hist(data = df, x = 'adults', bins = bin_edges)

plt.xlim(0, 5) # could also be called as plt.xlim((0, 35))
df['arrival_date_week_number'].describe()
np.log10(df['arrival_date_week_number']).describe()
plt.figure(figsize = [10, 5])



# left histogram: data plotted in natural units

plt.subplot(1, 2, 1)

bin_edges = np.arange(1, df['arrival_date_week_number'].max()+1, 1)

plt.hist(data=df,x='arrival_date_week_number', bins = bin_edges)

plt.xlabel('arrival_date_week_number')



# right histogram: data plotted after direct log transformation

plt.subplot(1, 2, 2)

log_data = np.log10(df['arrival_date_week_number']) # direct data transform

log_bin_edges = np.arange(0, log_data.max()+0.07, 0.07)

plt.hist(log_data, bins = log_bin_edges)

plt.xlabel('log(arrival_date_week_number)')
#sns.scatterplot(df['adults'],df['babies'])

df.corr()
rm=sm.OLS(df['required_car_parking_spaces'],df[['intercept','adults','children']])

result=rm.fit()

result.summary()