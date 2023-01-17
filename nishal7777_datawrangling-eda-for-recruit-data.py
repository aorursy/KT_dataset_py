"""DataFrame LookUp table

--------------------------------------------------------------------------------------------------

      DATAFRAME      |   Description                                                              |

++++++++++++++++++++ + +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

df_air_visit_data    | This contains visiting data of customers in 

                     | air system; Total 829 unique restaurants are visited.

                     | columns : air_store_id; visit_date; visitors

++++++++++++++++++++ + ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

df_air_store_info    | This contains restaurants(stores) info

                     | contained in air system;

                     | columns : air_store_id; air_genre_name(type of food being served);

                     |           air_area_name;latitude; longitude

++++++++++++++++++++ + +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

df_hpg_store_info    | This contains restaurants(stores) info

                     | contained in hpg system;

                     | columns : hpg_store_id; hpg_genre_name(type of food being served);

                     |           hpg_area_name;latitude; longitude

++++++++++++++++++++ + ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

df_air_reserve       | This contains visiting data of customers who have done reservation 

                     | air system;

                     | columns : air_store_id; visit_datetime; reserve_datetime; reserve_visitors

++++++++++++++++++++ + ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

df_hpg_reserve       | This contains visiting data of customers who have done reservation 

                     | hpg system;

                     | columns : hpg_store_id; visit_datetime; reserve_datetime; reserve_visitors

++++++++++++++++++++ + ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

df_store_id_realation| This dataframe contains mapping between air & hpg restaurants

                     |  It can be potentially used to join/merge the remaining dataframes

                     | columns : air_store_id; hpg_store_id

++++++++++++++++++++ + ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

df_date_info         |  It contains weekday and holiday information for a given calendar date

                     | columns : air_store_id; hpg_store_id

++++++++++++++++++++ + ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

df_test_set          | This is a test data having air store ids and calendar date and blank colulmn

                     |  to predict visitor count for each air_store_id

                     | columns : id, visitors

++++++++++++++++++++ + ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



++++++++++++++++++++ + ++ + +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

                          |

df_hpg_reserve_air_mapped | This is a inner join between df_hpg_reserve,df_store_id_realation

                          |  to get only those hpg stores which are having air store id

                          | Columns : visit_datetime; reserve_datetime; reserve_visitors; air_store_id

                          |           visit_date; holiday_flg; day_of_week

++++++++++++++++++++ + ++ + +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++   

df_hpg_reserve_air_total  | It contains total air & hpg reserve info, where hpg store id are mapped to

                          |  their air store ids.

                          |   It has information of 333 stores which are reserved to visit.

                          | Columns : air_store_id; day_of_week; holiday_flg; reserve_datetime

                                      reserve_visitors; visit_date; visit_datetime; time_diff

++++++++++++++++++++ + ++++ + +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

df_air_hpg_total_store_info |  It contains total air & hpg store info, where hpg store id are mapped to

                            |  their air store ids.

                            |  It has information of 892 stores info.

                            | Columns: air_area_name; air_genre_name; air_store_id; latitude; longitude

++++++++++++++++++++ + ++++ + +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

df_air_visit_train          |  This contains the df_air_visit_data, where holiday flag, weather inforamtion

                            |  is added along with.

                            | Columns : air_store_id; air_area_name; air_genre_name; latitude; longitude;

                            | visit_date; visitors; holiday_flg; day_of_week;

                            """

#Load all the data to dataFrames

import pandas as pd

import numpy as np

import re

import numpy as np

import matplotlib.pyplot as plt 

import matplotlib.dates as dates

import seaborn as sns

plt.style.use('fivethirtyeight')

#Create  dataframes

df_air_visit_data = pd.read_csv('../input/air_visit_data.csv') #this is the training data

df_air_store_info = pd.read_csv('../input/air_store_info.csv')

df_hpg_store_info = pd.read_csv('../input/hpg_store_info.csv')

df_air_reserve = pd.read_csv('../input/air_reserve.csv')

df_hpg_reserve = pd.read_csv('../input/hpg_reserve.csv')

df_store_id_realation = pd.read_csv('../input/store_id_relation.csv')

df_test_set = pd.read_csv('../input/sample_submission.csv')

df_date_info = pd.read_csv('../input/date_info.csv')
"""

Now inner join hpg_reserve, store_id_realation to get only those hpg stores which are having reservation info



This can be helpful in getting hpg to air relation too.

"""

df_hpg_reserve_air_mapped = pd.merge(df_hpg_reserve,df_store_id_realation,on='hpg_store_id',how='inner')
#Neat Date-time format conversion

df_air_reserve.visit_datetime = pd.to_datetime(df_air_reserve.visit_datetime)

df_air_reserve.reserve_datetime = pd.to_datetime(df_air_reserve.reserve_datetime)

df_air_visit_data.visit_date = pd.to_datetime(df_air_visit_data.visit_date)

df_hpg_reserve_air_mapped.visit_datetime = pd.to_datetime(df_hpg_reserve_air_mapped.visit_datetime)

df_hpg_reserve_air_mapped.reserve_datetime  = pd.to_datetime(df_hpg_reserve_air_mapped.reserve_datetime)

df_date_info.calendar_date = pd.to_datetime(df_date_info.calendar_date)
#change the index of df_date_info to calendar_date

df_date_info = df_date_info.set_index('calendar_date')
#creating holiday flag,,weekday column for air_reserve data

df_air_reserve['visit_date'] = df_air_reserve.visit_datetime.dt.date

df_air_reserve['holiday_flg'] = df_air_reserve.visit_date.apply(lambda x : df_date_info.loc[x]['holiday_flg'])

df_air_reserve['day_of_week'] = df_air_reserve.visit_date.apply(lambda x : df_date_info.loc[x]['day_of_week'])
#creating holiday flag ,weekday column for air_visit data

df_air_visit_data['holiday_flg'] = df_air_visit_data.visit_date.apply(lambda x : df_date_info.loc[x]['holiday_flg'])

df_air_visit_data['day_of_week'] = df_air_visit_data.visit_date.apply(lambda x : df_date_info.loc[x]['day_of_week'])
#creating holiday flag ,weekday column for df_hpg_reserve_air_mapped

df_hpg_reserve_air_mapped['visit_date'] = df_hpg_reserve_air_mapped.visit_datetime.dt.date

df_hpg_reserve_air_mapped['holiday_flg'] = df_hpg_reserve_air_mapped.visit_date.apply(lambda x : df_date_info.loc[x]['holiday_flg'])

df_hpg_reserve_air_mapped['day_of_week'] = df_hpg_reserve_air_mapped.visit_date.apply(lambda x : df_date_info.loc[x]['day_of_week'])
#Combine air_reserve and hpg_reserve data

df_hpg_reserve_air_mapped = df_hpg_reserve_air_mapped.drop('hpg_store_id',axis=1)

df_hpg_reserve_air_total = df_hpg_reserve_air_mapped.append(df_air_reserve)



#After appending, df_hpg_reserve_air_mapped with df_air_reserve, drop the duplicates 

df_hpg_reserve_air_total = df_hpg_reserve_air_total.drop_duplicates()
len(df_hpg_reserve_air_total.air_store_id.unique())
#Inner join df_hpg_store_info & df_store_id_realtion to get only required hpg stores which are mapped to air stores

df_hpg_store_info_joined = pd.merge(df_hpg_store_info,df_store_id_realation,how='inner',on='hpg_store_id')

df_hpg_store_info_joined = df_hpg_store_info_joined.drop('hpg_store_id',axis=1)
#Rename columns

df_hpg_store_info_joined = df_hpg_store_info_joined.rename(columns={'hpg_genre_name':'air_genre_name','hpg_area_name':'air_area_name'})
df_hpg_store_info_joined.head()
df_hpg_store_info_joined.shape
df_air_store_info.head()
df_air_hpg_total_store_info = df_air_store_info.append(df_hpg_store_info_joined)
df_air_hpg_total_store_info.head()
df_air_hpg_total_store_info.shape
# df_hpg_reserve_air_total --> dataFrame Having both hpg & air restaurant's reserve visitor's data

# df_air_hpg_total_store_info ---> dataFrame Having both hpg & air restaurant's store info, summing up 892 stores info
df_hpg_reserve_air_mapped.head()
df_hpg_reserve_air_total.head()
df_air_hpg_total_store_info.head()
#set index of df_air_hpg_total_store_info as air_store_id

#df_air_hpg_total_store_info = df_air_hpg_total_store_info.set_index('air_store_id')

df_air_hpg_total_store_info.head()
#Merge df_air_hpg_total_store_info, df_air_visit_data to include latitude, longitude, genre, location

df_air_visit_train = pd.merge(df_air_hpg_total_store_info,df_air_visit_data,on='air_store_id',how='inner')

df_air_visit_train.head()
#Adding time difference between reservation time & visiting time as a feature

df_hpg_reserve_air_total['time_diff'] = df_hpg_reserve_air_total['visit_datetime'].dt.date - df_hpg_reserve_air_total['reserve_datetime'].dt.date
df_hpg_reserve_air_total = pd.merge(df_air_hpg_total_store_info,df_hpg_reserve_air_total,on='air_store_id',how='inner')
df_air_hpg_total_store_info.head()
df_hpg_reserve_air_total.head()
df_hpg_reserve_air_total.groupby(['holiday_flg','day_of_week']).sum()

#for holiday_flg = 0 (not a holiday) has more reserved visitors when compared to that of holiday(holiday_flg=1) for each weekday

#It means there is a good amount of business in weekdays which aren't holidays.

#interesting thing is though it's holiday or not; friday has highest number of customers in a week
#change the type of holiday_flg, day_of_week 

df_hpg_reserve_air_total.holiday_flg = df_hpg_reserve_air_total.holiday_flg.astype('category')

df_hpg_reserve_air_total.day_of_week = df_hpg_reserve_air_total.day_of_week.astype('category')
#do the short form of calendar days, viz Firday : Fri

df_hpg_reserve_air_total.day_of_week = df_hpg_reserve_air_total.day_of_week.map({"Friday":"Fri","Saturday":"Sat","Sunday":"Sun","Wednesday":"Wed","Monday":"Mon","Tuesday":"Tue","Thursday":"Thu"})
sns.set(style='whitegrid')

sns.factorplot(data=df_hpg_reserve_air_total, x='day_of_week',y='reserve_visitors',hue='holiday_flg',kind='box',size=10)

plt.title('Box Plot of holiday trend in reserve visitors')

sns.factorplot(data=df_hpg_reserve_air_total, x='day_of_week',y='reserve_visitors',hue='holiday_flg',kind='bar',size=10)

plt.title('Bar Plot holiday trend in reserve visitors')

plt.show()
sns.pointplot(data=df_hpg_reserve_air_total, x='day_of_week',y='reserve_visitors',hue='holiday_flg',

             palette = {0:"g",1:"m"},

              markers=["^", "o"], linestyles=["-", "--"]

             )

plt.show()
df_air_visit_train.groupby(['holiday_flg','day_of_week'])['visitors'].sum()
df_air_visit_train.holiday_flg = df_air_visit_train.holiday_flg.astype('category')

df_air_visit_train.day_of_week = df_air_visit_train.day_of_week.astype('category')

df_air_visit_train.day_of_week = df_air_visit_train.day_of_week.map({"Friday":"Fri","Saturday":"Sat","Sunday":"Sun","Wednesday":"Wed","Monday":"Mon","Tuesday":"Tue","Thursday":"Thu"})
sns.set(style='whitegrid')

sns.factorplot(data=df_air_visit_train, x='day_of_week',y='visitors',hue='holiday_flg',kind='box',size=10)

plt.title('Box Plot of holiday trend in  visitors with no reservations')

sns.factorplot(data=df_air_visit_train, x='day_of_week',y='visitors',hue='holiday_flg',kind='bar',size=10)

plt.title('Bar Plot holiday trend in  visitors with no reservations')

plt.show()
sns.pointplot(data=df_air_visit_train, x='day_of_week',y='visitors',hue='holiday_flg',

             palette = {0:"g",1:"m"},

              markers=["^", "o"], linestyles=["-", "--"]

             )

plt.show()
df1 = df_hpg_reserve_air_total[['visit_date', 'reserve_visitors']].groupby('visit_date').sum().reset_index()

#df1.visit_date = df1.visit_date.astype('str')

df2 = df_air_visit_train[['visit_date','visitors']].groupby('visit_date').sum().reset_index()

#df2.visit_date = df2.visit_date.astype('str')
df1 = df1.set_index('visit_date')

df2 = df2.set_index('visit_date')
df1.head()
df2.head()
f,ax=plt.subplots(1,1,figsize=(15,8))

df1.plot(color='c',kind='line',ax=ax)

df2.plot(color='r',kind='area',ax=ax)

plt.ylabel('Visitor Count')

plt.title('Trend in visitor count with and without reservations')

plt.show()
df_hpg_reserve_air_total.head()
df1_genre = df_air_visit_train[['air_genre_name','visitors']].groupby('air_genre_name').sum().reset_index()

df1_genre = df1_genre.set_index('air_genre_name')
f,ax=plt.subplots(1,1,figsize=(15,8))

ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)

df1_genre.plot(color='c',kind='bar',ax=ax)

plt.title("Trending Cuisine(No Reservations)")

plt.ylabel('Overall Restaurants Visitor Count')

plt.xlabel('Cuisine')

plt.show()
df2_genre = df_hpg_reserve_air_total[['air_genre_name','reserve_visitors']].groupby('air_genre_name').sum()
df2_genre
f,ax=plt.subplots(1,1,figsize=(15,8))

ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)

df2_genre.plot(color='c',kind='bar',ax=ax)

plt.title("Trending Cuisine(With reservations)")

plt.ylabel('Overall Restaurants Visitor Count')

plt.xlabel('Cuisine')

plt.show()
df_air_visit_train.head()
df1_restaurant = df_air_visit_train[['latitude','longitude','air_store_id']]

#dropping duplicates of weather location and air_store_ids

df1_restaurant = df1_restaurant.drop_duplicates()

df1_restaurant['location'] = df1_restaurant['latitude'] + df1_restaurant['longitude']
df1_restaurant = df1_restaurant.groupby('location').count().reset_index()
sns.regplot(data=df1_restaurant, x='location', y='air_store_id',fit_reg=False)

plt.title("Location tracking of most visited(without reservations) competitive restaurants")

plt.show()
df_hpg_reserve_air_total.head()
df2_restaurant = df_hpg_reserve_air_total[['latitude','longitude','air_store_id']]

#dropping duplicates of weather location and air_store_ids

df2_restaurant = df2_restaurant.drop_duplicates()

df2_restaurant['location'] = df2_restaurant['latitude'] + df2_restaurant['longitude']
df2_restaurant = df2_restaurant.groupby('location').count().reset_index()
sns.regplot(data=df2_restaurant, x='location', y='air_store_id',fit_reg=False)

plt.title("Location tracking of most reserved competitive restaurants")

plt.show()