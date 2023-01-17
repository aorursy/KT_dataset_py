import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
# read csv file using pandas
df = pd.read_csv('../input/temperature-timeseries-for-some-brazilian-cities/station_rio.csv')
df.head()
df = df.rename(columns={'metANN' : 'Temp', 'YEAR' : 'Year'})
df.head(20)
# checking null values on the original dataframe
df.isnull().values.any()
year = df['Year']
temp = df['Temp']

plt.hist(temp, bins = 10)
df.columns
# replacing inconsistent values (999.90): condition: values equal or greater then 50 ºC
# for all columns, except for 'Year', replace values above 50 ºC by null (np.nan)

for i in df.columns:
  if i != 'Year':
    df.loc[df[i] >= 50, i] = np.nan
# verifying changes (NaN)
df.loc[10:15]
# now we have null values on the dataframe
df.isnull().values.any()
# How many inconsistent values (greater than 50) did the dataframe have in each column?

empty_entries_per_column = df.isna().sum(axis = 0)
empty_entries_per_column
# How many inconsistent values (greater than 50) did the dataframe have in each row?

empty_entries_per_row = df.isna().sum(axis = 1)
empty_entries_per_row
# ploting Year x Average Temperature 

plt.bar(df['Year'], df['Temp'])
plt.ylim(23,26)
plt.title('Rio de Janeiro Average Temperature (1973 - 2019)')
# same plot, with zoom
plt.bar(df['Year'], df['Temp'])
plt.ylim(23,26)
plt.xlim(2000, 2020)
plt.xticks(ticks = np.arange(2000, 2020, step = 1), labels = np.arange(2000, 2020, step = 1), rotation = 90)
plt.title('Rio de Janeiro Average Temperature (2000 - 2019)')
# mean temperature (all years: 1973 - 2019)
avg_temp = round(df.Temp.mean(), 2)
avg_temp
# seasons mean temperatures
summer = df['D-J-F'].mean()
autumn = df['M-A-M'].mean()
winter = df['J-J-A'].mean()
spring = df['S-O-N'].mean()

round(summer, 2), round(autumn, 2), round(winter, 2), round(spring, 2)
# preparing data to categorical bar plot

data = {'Summer': round(summer, 2), 'Autumn': round(autumn, 2), 'Winter': round(winter, 2), 'Spring': round(spring, 2)}
names = list(data.keys())
values = list(data.values())
len(names)
# season mean temperature plot

bar1 = plt.bar(np.arange(len(values)), values)
plt.xticks(range(len(names)), names)
plt.title('Rio de Janeiro: Average Temperature for Each Season')
plt.ylim(20,30)
plt.axhline(avg_temp, color = 'r', linestyle = 'dashed')
for rect in bar1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.2f' % float(height), ha='center', va='bottom', fontsize = 12, fontweight = 'bold')
# Summer temperatures
plt.bar(df['Year'], df['D-J-F'], color = 'red')
plt.ylim(23,30)
plt.title('Summer temperatures (1973-2019)')
# Maximum summer temperature
hottest_summer = df['D-J-F'].max()
print("Hottest summer temperature (average):", hottest_summer, "ºC")

# index of maximum summer temperature
idx = df['D-J-F'].idxmax()

# Year of maximum summer temperature
df['Year'][idx]
print("Year of the hottest summer:", df['Year'][idx])
# Minimum summer temperature
coldest_summer = df['D-J-F'].min()
print("Coldest summer temperature (average):", coldest_summer, "ºC")

# index of maximum summer temperature
idx = df['D-J-F'].idxmin()

# Year of maximum summer temperature
df['Year'][idx]
print("Year of the coldest summer:", df['Year'][idx])
# Winter temperatures
plt.bar(df['Year'], df['J-J-A'], color = 'grey')
plt.ylim(18,25)
plt.title('Winter temperatures (1973-2019)')
# Maximum winter temperature
hottest_winter = df['J-J-A'].max()
print("Hottest winter temperature (average):", hottest_winter, "ºC")

# index of maximum winter temperature
idx = df['J-J-A'].idxmax()

# Year of maximum winter temperature
df['Year'][idx]
print("Year of the hottest winter:", df['Year'][idx])
# Minimum winter temperature
coldest_winter = df['J-J-A'].min()
print("Coldest winter temperature (average):", coldest_winter, "ºC")

# index of minimum winter temperature
idx = df['J-J-A'].idxmin()

# Year of minimum winter temperature
df['Year'][idx]
print("Year of the coldest winter:", df['Year'][idx])
# preparing data to categorical bar plot

data = {'Summer Max (2015)': round(hottest_summer, 2),
        'Summer Min (1979)': round(coldest_summer, 2), 
        'Winter Max (1995)': round(hottest_winter, 2), 
        'Winter Min (1988)': round(coldest_winter, 2)}

names = list(data.keys())
values = list(data.values())
plt.figure(figsize=(10,5))
bar2 = plt.bar(np.arange(len(values)), values)
plt.xticks(range(len(names)), names)
plt.title('Rio de Janeiro: Summer x Winter Extreme Temperatures (1973-2019)')
plt.ylim(15,30)
plt.axhline(avg_temp, color = 'r', linestyle = 'dashed')
for rect in bar2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.2f' % float(height), ha='center', va='bottom', fontsize = 12, fontweight = 'bold')