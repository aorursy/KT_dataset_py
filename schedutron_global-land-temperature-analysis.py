import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
%matplotlib inline
plt.style.use('ggplot')

months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
dataFrame = pd.read_csv('../input/GlobalTemperatures.csv')
dataFrame.head()
dataFrame.tail()
d1 = dataFrame.iloc[:,:2]  # Throwing off unnecessary data, for now.
d1.head()
data = d1['LandAverageTemperature']
mean_temp = data.mean()
variance = data.var()
std = data.std()
print('Mean temperature throughout the years: %.3f celsius' % mean_temp)
print('Temperature variance throughout the dataset: %.3f celsius squared' % variance)
print('Temperature standard deviation throughout the dataset: %.3f celsius' % std)
times = pd.DatetimeIndex(d1['dt'])
grouped = d1.groupby([times.year]).mean()
plt.figure(figsize= (20,10))
std_grouped = d1.groupby([times.year]).std()
plt.plot(grouped['LandAverageTemperature'])
plt.plot(std_grouped['LandAverageTemperature'])
plt.title('Average Land Temperature and STD between 1750 - 2015')
plt.xlabel('Year')
plt.ylabel('Temperature (Celsius)')
plt.legend(['Average Land Temperature', 'Standard Deviation in Temperature'])
plt.show()
grouped.head()
grouped.tail()
d1[times.year == 1752]
d2 = dataFrame[['dt', 'LandAverageTemperatureUncertainty']]
uncertainty_group = d2.groupby([times.year]).mean()
plt.figure(figsize= (20,10))
plt.plot(uncertainty_group['LandAverageTemperatureUncertainty'])
plt.title('Average Land Temperature Uncertainty between 1750 - 2015')
plt.xlabel('Year')
plt.ylabel('Temperature Measurement Uncertainty  (Celsius)')
plt.show()
d1['LandAverageTemperature']= d1['LandAverageTemperature'].fillna(method='ffill')
# Display coldest and warmest times
min_max_df = dataFrame[['dt', 'LandMaxTemperature', 'LandMinTemperature']]
warmest_time_idx = min_max_df['LandMaxTemperature'].idxmax()
warmest_time = min_max_df.loc[warmest_time_idx]['dt']
max_temp = min_max_df['LandMaxTemperature'].max()
print('Highest temperature recorded: %.3f celsius in %s %s' % (max_temp, months[int(warmest_time.split('-')[1])-1], warmest_time[:4]))

coldest_time_idx = min_max_df['LandMinTemperature'].idxmin()
coldest_time = min_max_df.loc[coldest_time_idx]['dt']
min_temp = min_max_df['LandMinTemperature'].min()
print('Lowest temperature recorded: %.3f celsius in %s %s' % (min_temp, months[int(coldest_time.split('-')[1])-1], coldest_time[:4]))
print('Temperature range: %.3f celsius' % (max_temp-min_temp))
# Display coldest and warmest years
warmest_year = grouped['LandAverageTemperature'].idxmax()
max_avg_temp = grouped.loc[warmest_year]['LandAverageTemperature']
print('\nWarmest year -> %s : %.3f celsius' % (warmest_year, max_avg_temp))


coldest_year = grouped['LandAverageTemperature'].idxmin()
min_avg_temp = grouped.loc[coldest_year]['LandAverageTemperature']
print('Coldest year -> %s : %.3f celsius' % (coldest_year, min_avg_temp))
# Histogram of temperatures grouped by month - of 1750, 2015 and average of all the years
# Draw the year's average temperature line
first_year_df = d1[:12]
first_year_df = first_year_df.assign(month=pd.Series(months))
first_year_df.set_index("month",drop=True,inplace=True)
first_year_df.rename(columns={'LandAverageTemperature': 'avg_temp_first'}, inplace=True)

final_year_df = d1[-12:]
final_year_df.index = list(range(12))
final_year_df = final_year_df.assign(month=pd.Series(months))
final_year_df.set_index("month",drop=True,inplace=True)
final_year_df.rename(columns={'LandAverageTemperature': 'avg_temp_final'}, inplace=True)

frames = [first_year_df, final_year_df]
temp_variation = pd.concat(frames, axis=1)
temp_variation['diff'] = temp_variation['avg_temp_final'] - temp_variation['avg_temp_first']
temp_variation
ax = plt.figure(figsize=(20,10)).gca()
temp_variation.plot(kind='bar', y=['avg_temp_first', 'avg_temp_final'], ax=ax)
temp_variation.plot(kind='line', y=['diff'], style='go-', ax=ax)
plt.title('Monthly average temperatures for years 1750 and 2015')
plt.xlabel('Month')
plt.ylabel('Average Temperature (Celsius)')
plt.legend(['difference', '1750 average', '2015 average'])
plt.show()
month_avg_group = d1.groupby([times.month]).mean()
month_avg_group.index = list(range(12))
month_avg_group = month_avg_group.assign(month=pd.Series(months))
month_avg_group.set_index("month",drop=True,inplace=True)

month_avg_group
warmest_month = month_avg_group['LandAverageTemperature'].idxmax()
coldest_month = month_avg_group['LandAverageTemperature'].idxmin()

print('Warmest month throughout the years -> %s : %.3f celsius average temperature' % (warmest_month, month_avg_group.loc[warmest_month]['LandAverageTemperature']))
print('Coldest month throughout the years -> %s : %.3f celsius average temperature' % (coldest_month, month_avg_group.loc[coldest_month]['LandAverageTemperature']))
ax = plt.figure(figsize=(20,10)).gca()
month_avg_group.plot(kind='bar', ax=ax)
plt.title('Monthly average temperatures throughout 1750-2015')
plt.xlabel('Month')
plt.ylabel('Average Temperature (Celsius)')
plt.show()
# Also maybe have running averages in plots
# plot variation in temperature throughout the years
# First, data preprocessing to group min, max, diff data
# Group by year
yearly_var_group = min_max_df.groupby([times.year]).mean()
yearly_var_group.rename(columns={'LandMaxTemperature': 'max_avg_temp'}, inplace=True)
yearly_var_group.rename(columns={'LandMinTemperature': 'min_avg_temp'}, inplace=True)

yearly_var_group['diff'] = yearly_var_group['max_avg_temp'] - yearly_var_group['min_avg_temp']

ax = plt.figure(figsize=(20,10)).gca()
yearly_var_group.plot(y=['max_avg_temp', 'min_avg_temp', 'diff'], ax=ax)
plt.title('Variation in average temperatures throughout the dataset')
plt.xlabel('Year')
plt.ylabel('Temperature (Celsius)')
plt.legend(['Highest Temperature', 'Lowest Temperature', 'Difference'])
plt.show()
# Which month shows the highest variation?
month_var_group = min_max_df.groupby([times.month]).mean()
#month_var_group = month_max_group.drop('dt', axis=1)
month_var_group.rename(columns={'LandMaxTemperature': 'max_temp'}, inplace=True)
month_var_group.rename(columns={'LandMinTemperature': 'min_temp'}, inplace=True)

month_var_group.index = list(range(12))

month_var_group = month_var_group.assign(month=pd.Series(months))
month_var_group.set_index("month",drop=True,inplace=True)
month_var_group['diff'] = month_var_group['max_temp'] - month_var_group['min_temp']
month_var_group
# Plot the results

ax = plt.figure(figsize=(20,10)).gca()
month_var_group.plot(kind='bar', y=['max_temp', 'min_temp'], ax=ax)
month_var_group.plot(kind='line', y=['diff'], style='go-', ax=ax)
month_avg_group.plot(kind='line', style='co-', linewidth=2.0, ax=ax)
plt.title('Highest and lowest temperatures of months throughout the dataset')
plt.xlabel('Month')
plt.ylabel('Variation in Average Temperature (Celsius)')
plt.legend(['Difference', 'Average Temperature', 'Highest Temperature', 'Lowest Temperature'])
plt.show()
print('Mean temperature difference: %.3f celsius' % (month_var_group['diff'].mean()))
# Frequency distribution - do a histogram-like plot.
max_avg_temp = data.max()
min_avg_temp = data.min()
avg_temp_range = max_avg_temp - min_avg_temp
print('Average temperature range: %.3f celsius' % avg_temp_range)

sns.set()
plt.figure(figsize= (20,10))
sns.distplot(data)
plt.title('Frequency Distribution of Average Temperatures')
plt.xlabel('Temperature (celsius)')
plt.ylabel('Frequency (normalized / relative)')
plt.show()
# Also, let's just do a simple box plot

plt.figure(figsize=(20, 10))
d1.boxplot(column='LandAverageTemperature')
plt.title('Average Temperatures Box Plot')
plt.ylabel('Temperature (celsius)')
plt.show()
from sklearn.linear_model import LinearRegression as LinReg
# Decomposing seasons out of temperature data
d1.index = times

res = sm.tsa.seasonal_decompose(d1['LandAverageTemperature'])
fig, (ax1,ax2,ax3) = plt.subplots(3, 1, figsize=(30, 20))
d1.plot(ax=ax1)
ax1.set(xlabel='Time', ylabel='Temperature (celsius)')

res.trend.plot(ax=ax2)
ax2.set(xlabel='Time', ylabel='Trend')

res.seasonal.plot(ax=ax3)
ax3.set(xlabel='Time', ylabel='Seasonal')

plt.show()
from sklearn.linear_model import LinearRegression as LinReg

x = grouped.index.values.reshape(-1,1)
y = grouped['LandAverageTemperature'].values

reg = LinReg()
reg.fit(x,y)
y_preds = reg.predict(x)
print("Accuracy: %.3f" % reg.score(x,y))
plt.figure(figsize = (20,10))
plt.title("Linear Regression")
plt.scatter(x = x, y = y_preds)
plt.scatter(x = x,y =y,c = "b")
plt.show()
print('Predicted temperature of 2300 is: %.3f celsius' % reg.predict([[2300]])[0])