# Linear Algebra Packages

import pandas as pd

import numpy as np

import datetime as dt

from datetime import datetime



# Visualization Packages

import seaborn as sns

import matplotlib.pyplot as plt

%pylab inline

import folium

from folium.plugins import HeatMap
df = pd.read_csv('../input/traffic-collisions-in-san-diego-since-2017/pd_collisions_datasd_v1.csv')
print('Total Number of Accidents in Database: {}'.format(df.shape[0]))
df.sample(3)
print('Records are between {} and {}'.format(df.date_time.min(), df.date_time.max()))
# Remove blanks in dataframe with NAN

df = df.replace(r'^\s*$', np.nan, regex=True)
df.info()
charge_type_df = pd.DataFrame(df['violation_section'].value_counts())



# Create a Violation Type Counts Dataframe

viosec_df = pd.DataFrame(df['violation_section'].value_counts())

viosec_df_top = charge_type_df.iloc[:25] # Limit our output top 15 results



# Barplot

plt.figure(figsize=(12,6))

ax1 = sns.barplot(data=viosec_df_top, x=viosec_df_top.index, y='violation_section', palette=("Reds_d"))



# Formatting

sns.set_context("paper")

plt.title('SD Traffic Collision Counts since 2015 by Violation Section Code')

plt.xlabel('Violation Code')

plt.ylabel('Number of Collisions')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(12,6))

viosec_df_top10 = viosec_df.iloc[:10] # Limit our output top 10 results

viosec_pie = plt.pie(data=viosec_df_top10, x='violation_section', autopct='%.0f%%', shadow=True)

plt.title('SD Traffic Collision Counts since 2015 by Violation (Top 10 Violation Codes)')

plt.legend(viosec_df.index, loc=(1,0.2))

plt.show()
# List the unique violations

unique_violations = df.violation_section.unique()

totalViolations = []



# Count each unique violation

for violation in unique_violations:

    deadly_vio_df = df[df.violation_section==violation]

    totalViolations.append((violation, sum(deadly_vio_df.injured), sum(deadly_vio_df.killed)))

    

# Sort and determine top 10 violations by injury and by fatalities

top10_vio_sort_injury = sorted(totalViolations, key=lambda tup: tup[1], reverse = True)[:10]

top10_vio_sort_killed = sorted(totalViolations, key=lambda tup: tup[2], reverse = True)[:10]

top10vio_injury_df = pd.DataFrame(data=top10_vio_sort_injury, columns = ['violation','injured','killed'])

top10vio_killed_df = pd.DataFrame(data=top10_vio_sort_killed, columns = ['violation','injured','killed'])
fig = plt.figure()



ax1 = fig.add_axes([0, 0, 1, 1], aspect=1)

plt.pie(x=top10vio_injury_df.injured, autopct='%.0f%%', shadow=True)

plt.legend(top10vio_injury_df.violation, loc=(1,0.3))



ax2 = fig.add_axes([1, 0, 1, 1], aspect=1)

plt.pie(x=top10vio_killed_df.killed, autopct='%.0f%%', shadow=True)

plt.legend(top10vio_killed_df.violation, loc=(1,0.3))



ax1.set_title('Worst Violations by Percentage of Total Injured')

ax2.set_title('Worst Violations by Percentage of Total Killed')

plt.show()
# Strip time from datetime string in our dataset

date_format = "%Y-%m-%d %H:%M:%S"

max_record = datetime.datetime.strptime(str(df.date_time.max()), date_format)

min_record = datetime.datetime.strptime(str(df.date_time.min()), date_format)



# Calculate number of days in our records to calculate mean

df.date_time = pd.to_datetime(df.date_time)

df["day_of_week"] = df.date_time.dt.dayofweek

delta = max_record - min_record

days_count = delta.days

days = list(range(7))

day_mean_df = pd.DataFrame(columns = ['day','mean'])

day_mean_df.day = ['Sun','Mon','Tues','Wed','Thurs','Fri','Sat']



def day_stats(day):

    day_mean = df[df['day_of_week']==day].shape[0]/days_count

    return(day_mean)

    

for day in days:

    day_mean_df.loc[day, 'mean'] = day_stats(day)



plt.figure(figsize=(12,6))

sns.barplot(data=day_mean_df, x='day', y='mean', palette=("Blues_d"))



plt.title('SD Traffic Collision Counts since 2015 by Day of the Week (mean)')

plt.xlabel('Day of the Week')

plt.ylabel('Mean Number of Collisions')

plt.show()
# Mean and Std. Dev by Day

df.index = pd.DatetimeIndex(df.date_time)

day_stats_df = pd.DataFrame(df.resample('D').size())

day_stats_df['day_mean'] = df.resample('D').size().mean()

day_stats_df['day_std'] = df.resample('D').size().std()



# Upper and Lower Control Limit

UCL = day_stats_df['day_mean'] + 3 * day_stats_df['day_std']

LCL = day_stats_df['day_mean'] - 3 * day_stats_df['day_std']



# Plot

plt.figure(figsize=(15,6))

df.resample('D').size().plot(label='Accidents per day', color='sandybrown')

UCL.plot(color='red', ls='--', linewidth=1.5, label='UCL')

LCL.plot(color='red', ls='--', linewidth=1.5, label='LCL')

day_stats_df['day_mean'].plot(color='red', linewidth=2, label='Average')

plt.title('Traffic Collisions Timeline', fontsize=16)

plt.xlabel('')

plt.ylabel('Number of Traffic Collisions')

plt.tick_params(labelsize=14)
day_stats_df.drop(day_stats_df.tail(1).index, inplace=True) # drop last 1 row (since the records are incomplete for that day)

worst_day = day_stats_df[0].idxmax()

best_day = day_stats_df[0].idxmin()

print ('Worst Day was {}: {} accidents and Best Day was {}: {} accidents'.format(worst_day, day_stats_df[0].max(), best_day, day_stats_df[0].min()))
# Count frequency of collisions by weekday / appending a month column

months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 

              'August', 'September', 'October', 'November', 'December']

df["accident_month"] = df.date_time.dt.month

month_df = pd.DataFrame(df['accident_month'].value_counts()).sort_index()

month_df['month_name'] = months

month_df.columns = ['accident_count', 'month']



# Re-order columns

columnsTitles=["month","accident_count"]

month_df=month_df.reindex(columns=columnsTitles)



sns.barplot(data=month_df, x='month', y='accident_count', palette=("Reds_d"))



plt.title('SD Traffic Collision Counts since 2015 by Month')

plt.xlabel('Month')

plt.ylabel('Mean Number of Collisions')

plt.xticks(rotation=90)

plt.show()
month1_df = month_df[0:9]

month1_avg_df = pd.DataFrame(month1_df['accident_count'].div(3))



month2_df = month_df[9:12]

month2_avg_df = pd.DataFrame(month2_df['accident_count'].div(2))



merged_df = pd.merge(month1_avg_df, month2_avg_df, how='outer')

merged_df['month'] = months

merged_df



columnsTitles=["month","accident_count"]

merged_df=merged_df.reindex(columns=columnsTitles)



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,6))

fig.suptitle("SD Traffic Collisions Frequency by Month", fontsize=16)



ax4a = sns.barplot(x='month', y='accident_count', color='Red', data=merged_df, ax = axes[0])

ax4a.set(xlabel='Month', ylabel='Number of Accidents')



ax4b = plt.pie(x=merged_df.accident_count, labels=months, colors=['red', 'darkorange', 'silver'], autopct='%.0f%%', shadow=True)

plt.show()
date_before = datetime.date(2019, 10, 1)

df2 = df[df.date_time < date_before]



month_line_df = df2.resample('M').size()

plt.figure(figsize=(15,6))

month_line_df.plot(label='Total,  accidents per month', color='sandybrown')

month_line_df.rolling(window=12).mean().plot(color='red', linewidth=5)

plt.title('SD Total Traffic Collisions Per Month', fontsize=16)

plt.xlabel('')

plt.show()
print("Best Month {0}: {1} accidents".format(month_line_df.idxmin(), month_line_df[month_line_df.idxmin()]))

print("Worst Month {0}: {1} accidents".format(month_line_df.idxmax(), month_line_df[month_line_df.idxmax()]))
winter_df = pd.DataFrame(merged_df.iloc[[0,1,11],[1]].sum())

spring_df = pd.DataFrame(merged_df.iloc[[2,3,4],[1]].sum())

summer_df = pd.DataFrame(merged_df.iloc[[5,6,7],[1]].sum())

fall_df = pd.DataFrame(merged_df.iloc[[8,9,10],[1]].sum())



season1_df = pd.merge(winter_df, spring_df, how='outer')

season2_df = pd.merge(summer_df, fall_df, how='outer')

season_df = pd.merge(season1_df, season2_df, how='outer')

season_df['season'] = ['winter', 'spring', 'summer', 'fall']

season_df.rename(columns={0:'accident_count'}, inplace=True)

season_df = season_df.reindex(columns=['season','accident_count'])



plt.figure(figsize=(12,6))

ax5 = sns.barplot(x='season', y='accident_count', color='coral', data=season_df)

ax5.set(xlabel='Season of the Year', ylabel='Number of Accidents')

plt.show()
df["hour"] = df.date_time.dt.hour

print('The Mean Hour for Traffic Collisions: {number:.{digits}f}'.format(number=df.hour.mean(), digits=2))

print('The Median Hour for Traffic Collisions: {}'.format(df.hour.median()))

print('The Standard Deviation for Traffic Collisions: {number:.{digits}f}'.format(number=df.hour.std(), digits=2))

df.hour.describe()
plt.figure(figsize=(10,4))

sns.catplot(x='hour', kind='count',height=8.27, aspect=3, color='black',data=df)

plt.show()
plt.figure(figsize=(15,6))

df["hour"] = df.date_time.dt.hour

day_dict = {0:'SUN', 1:'MON', 2:'TUES', 3:'WED', 4:'THURS', 5:'FRI', 6:'SAT'}

df['day_of_week_name'] = df['day_of_week'].map(day_dict)

day_time_bplot = sns.boxplot(y='hour', x='day_of_week_name', data=df, width=0.5, palette='colorblind')

plt.xlabel("Day of the Week")

plt.ylabel("Hour of the Day")

plt.title("Boxplot of Traffic Collisions (Day of Week to Hour of Day)")

plt.show()
plt.figure(figsize=(15,6))

day_time_vioplot = sns.violinplot(x='day_of_week_name', y='hour', data=df)

plt.xlabel("Day of the Week")

plt.ylabel("Hour of the Day")

plt.title("Violinplot of Traffic Collisions (Day of Week to Hour of Day)")

plt.show()
# Replace NAN with empty string

df1 = df.replace(np.nan, '', regex=True)

df1["address_number_primary"]= df1["address_number_primary"].astype(str)

df1['full_address'] = pd.DataFrame(df1['address_number_primary']+' '+df1['address_pd_primary']+' '+df1['address_road_primary']+' '+df1['address_sfx_primary']+', SAN DIEGO, CALIFORNIA, USA')



geo_df = pd.read_csv('../input/traffic-collisions-in-san-diego-since-2017/geocoded_pd_collisions_dataset.csv')
# Create basic Folium collision map

collision_map = folium.Map(location=[32.7167, -117.1661],

                       tiles = "Stamen Toner",

                      zoom_start = 10.2)



# Add data for heatmap 

geo_heatmap = geo_df[['latitude','longitude']]

geo_heatmap = geo_df.dropna(axis=0, subset=['latitude','longitude'])

geo_heatmap = [[row['latitude'],row['longitude']] for index, row in geo_heatmap.iterrows()]

HeatMap(geo_heatmap, radius=10).add_to(collision_map)

collision_map
goldentri = df[df['police_beat'].isin(['115', '931'])] # Includes 931, sorrento valley, since police officers often respond from this beat

goldentri_inj = goldentri[goldentri['injured']>1]
goldentri_charge = pd.DataFrame(goldentri_inj['violation_section'].value_counts())

golden_top = goldentri_charge.iloc[:25] # Limit our output top 15 results

                                   

plt.figure(figsize=(12,6))

ax1 = sns.barplot(data=golden_top, x=golden_top.index, y='violation_section', palette=("Reds_d"))



# Formatting

sns.set_context("paper")

plt.title('SD Traffic Collisions (with More than One Injury) since 2015 in the Golden Triangle Area')

plt.xlabel('Violation')

plt.ylabel('Number of Collisions')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(10,4))

sns.catplot(x='hour', kind='count',height=8.27, aspect=3, color='black',data=goldentri)

plt.show()