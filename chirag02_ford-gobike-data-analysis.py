import pandas as pd

import numpy as np

import seaborn as sb

import matplotlib.pyplot as plt



%matplotlib inline
df = pd.read_csv('../input/201902-fordgobike-tripdata.csv')

df.head()
df.info()
#changing data type of start_time and end_time to datetime.

df.start_time = pd.to_datetime(df.start_time)

df.end_time = pd.to_datetime(df.end_time)
df.bike_share_for_all_trip = (df.bike_share_for_all_trip == 'Yes')
df.info()
df.describe()
binsize = 500

bins = np.arange(0, df['duration_sec'].max()+binsize, binsize)



plt.figure(figsize=[8, 5])

plt.hist(data = df, x = 'duration_sec', bins = bins)

plt.title('Distribution of Trip Durations')

plt.xlabel('Duration (sec)')

plt.ylabel('Number of Trips')

plt.axis([-500, 10000, 0, 90000])

plt.show()
log_binsize = 0.05

log_bins = 10 ** np.arange(2.4, np.log10(df['duration_sec'].max()) + log_binsize, log_binsize)



plt.figure(figsize=[8, 5])

plt.hist(data = df, x = 'duration_sec', bins = log_bins)

plt.title('Distribution of Trip Durations')

plt.xlabel('Duration (sec)')

plt.ylabel('Number of Trips')

plt.xscale('log')

plt.xticks([500, 1e3, 2e3, 5e3, 1e4], [500, '1k', '2k', '5k', '10k'])

plt.axis([0, 10000, 0, 15000])

plt.show()
# Plotting start station id distribution.

binsize = 1

bins = np.arange(0, df['start_station_id'].astype(float).max()+binsize, binsize)



plt.figure(figsize=[20, 8])

plt.xticks(range(0, 401, 10))

plt.hist(data = df.dropna(), x = 'start_station_id', bins = bins)

plt.title('Distribution of Start Stations')

plt.xlabel('Start Station')

plt.ylabel('Number of Stations')

plt.show()
# Plotting end station id distribution.

binsize = 1

bins = np.arange(0, df['end_station_id'].astype(float).max()+binsize, binsize)



plt.figure(figsize=[20, 8])

plt.xticks(range(0, 401, 10))

plt.hist(data = df.dropna(), x = 'end_station_id', bins = bins)

plt.title('Distribution of End Stations')

plt.xlabel('End Station')

plt.ylabel('Number of Stations')

plt.show()
# Plotting age distribution derived from member's birth year.

binsize = 1

bins = np.arange(0, df['member_birth_year'].astype(float).max()+binsize, binsize)



plt.figure(figsize=[8, 5])

plt.hist(data = df.dropna(), x = 'member_birth_year', bins = bins)

plt.axis([1939, 2009, 0, 12000])

plt.xticks([1939, 1949, 1959, 1969, 1979, 1989, 1999, 2009], [(2019-1939), (2019-1949), (2019-1959), (2019-1969), (2019-1979), (2019-1989), (2019-1999), (2019-2009)])

plt.gca().invert_xaxis()

plt.title('Distribution of User Age')

plt.xlabel('Age (years)')

plt.ylabel('Number of Users')

plt.show()
# plotting types of users on bar.

plt.figure(figsize=[8,5])

plt.bar(x = df.user_type.value_counts().keys(), height = df.user_type.value_counts() )

plt.xlabel('User Type')

plt.ylabel('Number of Users')

plt.show()
# plotting genders on bar.

plt.figure(figsize=[8,5])

plt.bar(x = df.member_gender.value_counts().keys(), height = df.member_gender.value_counts() )

plt.xlabel('Gender')

plt.ylabel('Number of Users')

plt.show()
plt.figure(figsize=[8,5])

plt.scatter((2019 - df['member_birth_year']), df['duration_sec'], alpha = 0.25, marker = '.' )

plt.axis([-5, 145, 500, 10500])

plt.xlabel('Age (years)')

plt.ylabel('Duaration (sec)')

plt.show()
plt.figure(figsize=[12,5])



plt.subplot(1, 2, 1)

plt.scatter((2019 - df['member_birth_year']), df['duration_sec'], alpha = 0.25, marker = '.' )

plt.axis([-5, 85, 500, 6500])

plt.xlabel('Age (years)')

plt.ylabel('Duaration (sec)')



plt.subplot(1, 2, 2)

bins_y = np.arange(500, 6500+1, 1000)

bins_x = np.arange(-5, 85+1, 10)

plt.hist2d((2019 - df['member_birth_year']), df['duration_sec'],

           bins = [bins_x, bins_y])

plt.colorbar(ticks=[10000, 20000, 30000, 40000]);

plt.show()
sorted(df.start_station_id.unique())
t = []



all_start_station_ids = sorted(df.start_station_id.unique())

for x in all_start_station_ids :

    t.append(df[df.start_station_id == x].duration_sec.sum()) 

total_duration = pd.Series(t)
plt.figure(figsize = [20, 8])

sb.lineplot(x = df['start_station_id'], y = total_duration)

plt.xticks(range(0, 401, 10))

plt.xlabel('Start Station')

plt.ylabel('Total Duration')

plt.show()
t = []



all_end_station_ids = sorted(df.end_station_id.unique())

for x in all_end_station_ids :

    t.append(df[df.end_station_id == x].duration_sec.sum()) 

total_duration = pd.Series(t)
plt.figure(figsize = [20, 8])

sb.lineplot(x = df['start_station_id'], y = total_duration)

plt.xticks(range(0, 401, 10))

plt.xlabel('End Station')

plt.ylabel('Total Duration')

plt.show()
plt.figure(figsize = [8, 5])

base_color = sb.color_palette()[1]

sb.boxplot(data = df, x = 'member_gender', y = 'duration_sec', color = base_color)

plt.xlabel('Gender')

plt.ylabel('Duration (sec)')

plt.show()
plt.figure(figsize = [8, 5])

base_color = sb.color_palette()[1]

sb.boxplot(data = df, x = 'member_gender', y = 'duration_sec', color = base_color)

plt.ylim([-10, 2000])

plt.xlabel('Gender')

plt.ylabel('Duration (sec)')

plt.show()
plt.figure(figsize = [8, 5])

base_color = sb.color_palette()[1]

sb.boxplot(data = df, x = 'user_type', y = 'duration_sec', color = base_color)

plt.xlabel('User Type')

plt.ylabel('Duration (sec)')

plt.show()
plt.figure(figsize = [8, 5])

base_color = sb.color_palette()[1]

sb.boxplot(data = df, x = 'user_type', y = 'duration_sec', color = base_color)

plt.ylim([-10, 2500])

plt.xlabel('User Type')

plt.ylabel('Duration (sec)')

plt.show()
gender_markers = [['Male', 's'],['Female', 'o'],['Other', 'v']]



for gender, marker in gender_markers:

    df_gender = df[df['member_gender'] == gender]

    plt.scatter((2019 - df_gender['member_birth_year']), df_gender['duration_sec'], marker = marker, alpha=0.25)

plt.legend(['Male','Female','Other'])

plt.axis([10, 80, -500, 9000 ])

plt.xlabel('Age (year)')

plt.ylabel('Duration (sec)')

plt.show()
df['age'] = (2019 - df['member_birth_year'])

genders = sb.FacetGrid(data = df, col = 'member_gender', col_wrap = 2, size = 5,

                 xlim = [10, 80], ylim = [-500, 9000])

genders.map(plt.scatter, 'age', 'duration_sec', alpha=0.25)

genders.set_xlabels('Age (year)')

genders.set_ylabels('Duration (sec)')



plt.show()
user_type_markers = [['Customer', 's'],['Subscriber', 'o']]



for utype, marker in user_type_markers:

    df_utype = df[df['user_type'] == utype]

    plt.scatter((2019 - df_utype['member_birth_year']), df_utype['duration_sec'], marker = marker, alpha=0.25)

plt.legend(['Customer','Subscriber'])

plt.axis([10, 80, -500, 9000 ])

plt.xlabel('Age (year)')

plt.ylabel('Duration (sec)')

plt.show()
user_types = sb.FacetGrid(data = df, col = 'user_type', col_wrap = 2, size = 5,

                 xlim = [10, 80], ylim = [-500, 9000])

user_types.map(plt.scatter, 'age', 'duration_sec', alpha=0.25)

user_types.set_xlabels('Age (year)')

user_types.set_ylabels('Duration (sec)')



plt.show()