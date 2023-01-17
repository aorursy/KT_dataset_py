# Import packages that for Data Analytics.
import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import polyfit, poly1d
# Import CSVs into pandas DataFrame.
gdp = pd.read_csv ('../input/gdp-per-capita.csv')
work_hours = pd.read_csv('../input/work-hours-per-week.csv', sep = None, engine='python')
alcohol_consumption = pd.read_csv('../input/alcohol-consumption.csv', sep = None, engine='python')
food_consumption = pd.read_csv('../input/food-consumption.csv', sep = None, engine='python')
life_expectancy = pd.read_csv('../input/life-expectancy.csv')
# Step 1 - Print the first 5 rows of the dataset for a preview.

# Remove the hash (#) from the following comments so it will turn into a code to print preview of a particular data set.
# It can only print one data set preview at a time, whichever the latest.

#gdp.head()
#work_hours.head()
#alcohol_consumption.head()
#food_consumption.head()
#life_expectancy.head()
# Step 2 -Assign 'years' as headers of columns and 'countries' as indexes.

# From the preview, we can see the indexes are still using number of positions and doesn't have consistent name.

# Assign the first column with countries names as index.
gdp.set_index("Country", inplace=True)
work_hours.set_index("Working hours per week", inplace=True)
alcohol_consumption.set_index("Unnamed: 0", inplace=True)
food_consumption.set_index("Unnamed: 0", inplace=True)
life_expectancy.set_index("Country", inplace=True)

# Update the name of headers of indexes, so every index header will be called 'Country'
work_hours.index.rename('Country', inplace=True)
alcohol_consumption.index.rename('Country', inplace=True)
food_consumption.index.rename('Country', inplace=True)


# Step 3 - Remove the data from before 1990.

# I create new copies and kept the original imported data sets.
gdp_recent = gdp.loc[:,'1990':]
work_hours_recent = work_hours.loc[:,'1990':]
alcohol_consumption_recent = alcohol_consumption.loc[:,'1990':]
food_consumption_recent = food_consumption.loc[:,'1990':]
life_expectancy_recent = life_expectancy.loc[:,'1990':]

# Step 4 - Remove the columns and or rows that don't have any content.

gdp_cleandata = gdp_recent.dropna(axis=0, how='all')
gdp_cleandata = gdp_cleandata.dropna(axis=1, how='all')

work_hours_cleandata = work_hours_recent.dropna(axis=0, how='all')
work_hours_cleandata = work_hours_cleandata.dropna(axis=1, how='all')

alcohol_consumption_cleandata = alcohol_consumption_recent.dropna(axis=0, how='all')
alcohol_consumption_cleandata = alcohol_consumption_cleandata.dropna(axis=1, how='all')

food_consumption_cleandata = food_consumption_recent.dropna(axis=0, how='all')
food_consumption_cleandata = food_consumption_cleandata.dropna(axis=1, how='all')

life_expectancy_cleandata = life_expectancy_recent.dropna(axis=0, how='all')
life_expectancy_cleandata = life_expectancy_cleandata.dropna(axis=1, how='all')

# Step 5 - Assign the data to float data type.

gdp_df = pd.DataFrame(data=gdp_cleandata, dtype=np.float).round(2)
work_hours_df = pd.DataFrame(data=work_hours_cleandata, dtype=np.float).round(2)
alcohol_consumption_df = pd.DataFrame(data=alcohol_consumption_cleandata, dtype=np.float).round(2)
food_consumption_df = pd.DataFrame(data=food_consumption_cleandata, dtype=np.float).round(2)
life_expectancy_df = pd.DataFrame(data=life_expectancy_cleandata, dtype=np.float).round(2)

# Preview the corrected data set.

# Remove the hash (#) from the following comments so it will turn into a code to print preview of a particular data set.
# It can only print one data set preview at a time, whichever the latest.

#gdp_df.head()
#work_hours_df.head()
#alcohol_consumption_df.head()
#food_consumption_df.head()
#life_expectancy_df.head()
#### Last year of available data.
print ('GDP data is available up to year ' + gdp_df.dtypes.index[-1])
print ('Work hours data is available up to year ' + work_hours_df.dtypes.index[-1])
print ('Alcohol consumption data is available up to year ' + alcohol_consumption_df.dtypes.index[-1])
print ('Food consumption data is available up to year ' + food_consumption_df.dtypes.index[-1])
print ('Life Expectancies data is available up to year ' + life_expectancy_df.dtypes.index[-1])
alcohol_consumption_df.count()
gdp_2005 = gdp_df.loc[:,'2005'].dropna()
work_hours_2005 = work_hours_df.loc[:,'2005'].dropna()
alcohol_consumption_2005 = alcohol_consumption_df.loc[:,'2005'].dropna()
food_consumption_2005 = food_consumption_df.loc[:,'2005'].dropna()
life_expectancy_2005 = life_expectancy_df.loc[:,'2005'].dropna()
# Remove the hash (#) from the following comments so it will turn into a code to print a particular series.
# It can only print one series at a time, whichever the latest.

#print (gdp_2005)
#print (work_hours_2005)
#print (alcohol_consumption_2005)
#print (food_consumption_2005)
#print (life_expectancy_2005)
def indicators_correlation (indicator1, indicator2):
    """
    This function takes two type float series to see how many data points of index 
    go to the same direction and different direction. This will give a quick overview of
    how those two series are correlated, either positively or negatively. 
    
    If the result gives one of the number extremely higher compare to another, it indicates strong correlation.
    If the result between two numbers are relatively equal, it indicates that the correlation is weak to no correlation.
    
    This formula is adapted from Udacity IPND lessons.
    """
    both_above = (indicator1 > indicator1.mean()) & (indicator2 > indicator2.mean())
    both_below = (indicator1 < indicator1.mean()) & (indicator2 < indicator2.mean())
    same_direction = both_above | both_below
    num_country_same_direction = same_direction.sum()
    num_country_diff_direction = (len(indicator1) - num_country_same_direction)
    return (num_country_same_direction,num_country_diff_direction)
# Removing data of countries without matches.
q1_pd = pd.concat([work_hours_2005, gdp_2005], axis=1).dropna()

# Update the name of headers. This will improve readability of the data.
q1_pd.columns.values[0] = 'Work Hours'
q1_pd.columns.values[1] = 'GDP'

# Re-split the data into 2 series for each indicator.
q1_work_hours = q1_pd.iloc[:,0]
q1_gdp = q1_pd.iloc[:,1]
q1_pd.describe()
x = q1_work_hours
sns.distplot(x, kde=False, fit=stats.gamma);

x = q1_gdp
sns.distplot(x, kde=False, fit=stats.gamma);
# Calculate how many countries that go to same direction and how many go their opposite way, using the function earlier.
indicators_correlation (q1_work_hours,q1_gdp)
q1_pd.corr()
gdp_2005.dropna().describe()
x = gdp_2005.dropna()
sns.distplot(x, kde=False, fit=stats.gamma);
# Copy the gdp_2005 series to a new series for the test. This is done so the original data set will not be modified.
gdp_2005_test = pd.DataFrame(data=gdp_2005, dtype=np.float).round(2)

# Create a new column to indicate where the gdp is less than the minimum GDP value we use for Q1. 
gdp_2005_test['lower_than_q1_gdp_min'] = np.where(gdp_2005_test<4887.37, True, False)

# Count how many low income countries that have been ignored in Work Hours data set.
gdp_2005_test['lower_than_q1_gdp_min'].sum()
# Create a new column to indicate where the gdp is higher than the maximum GDP value we use for Q1. 
gdp_2005_test['higher_than_q1_gdp_max'] = np.where(gdp_2005_test['2005']>51927.36, True, False)

# Count how many low income countries that have been ignored in Work Hours data set.
gdp_2005_test['higher_than_q1_gdp_max'].sum()
# Print a quick review to see if we did the classification correctly.
gdp_2005_test.head()
# Print a quick result to see if the data makes sense.
gdp_2005_test.loc[(gdp_2005_test['higher_than_q1_gdp_max'] == True)]
# Visualising the data and correlation using scatter plot. 
# The size of the bubbles are static and not influenced by any parameter. 
N = len(q1_pd)
x = q1_pd['Work Hours']
y = q1_pd['GDP']
colors = np.random.rand(N)
area = 15  # 0 to 15 point radius. It is static.

plt.scatter(x, y, s=area, c=colors, alpha=0.8)
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
plt.xlabel('Work Hours (/person/week)')
plt.ylabel('GDP (USD/capita/year)')
plt.title('Work Hours Vs GDP')
plt.grid(True)
plt.show()
q2_pd = pd.concat([work_hours_2005, alcohol_consumption_2005], axis=1).dropna()
q2_pd.columns.values[0] = 'Work Hours'
q2_pd.columns.values[1] = 'Alcohol Consumption'

q2_work_hours = q2_pd.iloc[:,0]
q2_alcohol_consumption = q2_pd.iloc[:,1]
q2_pd.describe()
x = q2_work_hours
sns.distplot(x, kde=False, fit=stats.gamma);
x = q2_alcohol_consumption
sns.distplot(x, kde=False, fit=stats.gamma);
# Print data below the mean.
q2_alcohol_consumption[(q2_alcohol_consumption <= q2_alcohol_consumption.mean())]
# Removing Turkey and it's data from the Q2 data.
q2_pd = q2_pd.drop('Turkey')

# Reslicing to get new series for both indicators.
q2_work_hours = q2_pd.iloc[:,0]
q2_alcohol_consumption = q2_pd.iloc[:,1]
q2_pd.describe()
# Calculate how many countries that go to same direction and how many go their opposite way, using the function earlier.
indicators_correlation (q2_work_hours,q2_alcohol_consumption)
q2_pd.corr()
alcohol_consumption_2005.dropna().describe()
x = alcohol_consumption_2005.dropna()
sns.distplot(x, kde=False, fit=stats.gamma);
# Copy the alcohol_consumption_2005 series to a new series for the test. 
# This is done so the original data set will not be modified.
alcohol_consumption_2005_test = pd.DataFrame(data=alcohol_consumption_2005, dtype=np.float).round(2)

# Create a new column to indicate where the alcohol consumption is less than the minimum consumption value we use for Q2. 
alcohol_consumption_2005_test['lower_than_q2_alc_consm_min'] = np.where(alcohol_consumption_2005_test<2.87, True, False)

# Count how many countries with low alcohol consumption have been ignored by merging with Work Hours data set.
alcohol_consumption_2005_test['lower_than_q2_alc_consm_min'].sum()
# Create a new column to indicate where the gdp is higher than the maximum GDP value we use for Q1. 
alcohol_consumption_2005_test['higher_than_q2_alc_consm_max'] = np.where(alcohol_consumption_2005_test['2005']>16.27, True, False)

# Count how many low income countries that have been ignored in Work Hours data set.
alcohol_consumption_2005_test['higher_than_q2_alc_consm_max'].sum()
# Print a quick review to see if we did the classification correctly.
alcohol_consumption_2005_test.head()
# Print a quick result to see if there is a notable pattern in the country that were excluded 
# due to minimum alcohol consumption.
alcohol_consumption_2005_test.loc[(alcohol_consumption_2005_test['lower_than_q2_alc_consm_min'] == True)]
# Print a quick result to see if the data makes sense.
alcohol_consumption_2005_test.loc[(alcohol_consumption_2005_test['higher_than_q2_alc_consm_max'] == True)]
# Visualising the data and correlation using scatter plot. 
# The size of the bubbles are static and not influenced by any parameter. 
N = len(q2_pd)
x = q2_pd['Work Hours']
y = q2_pd['Alcohol Consumption']
colors = np.random.rand(N)
area = 15  # 0 to 15 point radius. This is static.

plt.scatter(x, y, s=area, c=colors, alpha=0.8)
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
plt.xlabel('Work Hours (per person/week)')
plt.ylabel('Alcohol Consumption (litres/capita)')
plt.title('Work Hours Vs Alcohol Consumption')
plt.grid(True)
plt.show()
# Printing statistic properties of this serie before being processed further.
food_consumption_2005.describe()
# Printing statistic properties of this serie before being processed further.
alcohol_consumption_2005.describe()
# Only taking data for countries that have usable data in both series. 
q3_pd = pd.concat([food_consumption_2005, alcohol_consumption_2005], axis=1).dropna()
q3_pd.columns.values[0] = 'Food Consumption'
q3_pd.columns.values[1] = 'Alcohol Consumption'

q3_food_consumption = q3_pd.iloc[:,0]
q3_alcohol_consumption = q3_pd.iloc[:,1]
q3_pd.describe()
x = q3_food_consumption
sns.distplot(x, kde=False, fit=stats.gamma);
x = q3_alcohol_consumption
sns.distplot(x, kde=False, fit=stats.gamma);
# Visualising the data and correlation using scatter plot. 
# The size of the bubbles are static and not influenced by any parameter. 
N = len(q3_pd)
x = q3_pd['Alcohol Consumption']
y = q3_pd['Food Consumption']
colors = np.random.rand(N)
area = 5  # 0 to 15 point radius. This is static.

plt.scatter(x, y, s=area, c=colors, alpha=0.8)
plt.xlabel('Alcohol Consumption (litres/capita)')
plt.ylabel('Food Consumption (kcal/person/day)')
plt.title('Alcohol Consumption Vs Food Consumption')
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
plt.grid(True)
plt.show()
# Calculate how many countries that go to same direction and how many go their opposite way, using the function earlier.
indicators_correlation (q3_food_consumption,q3_alcohol_consumption)
q3_pd.corr()
len(gdp_2005)
len(life_expectancy_2005)
q4_df = pd.DataFrame(dict(food_consumption_2005 = food_consumption_2005, gdp_2005 = gdp_2005, \
                          life_expectancy_2005 = life_expectancy_2005)).dropna()
q4_df.describe()
# Plot normalised value (using Z-score normalization) of columns.
# For this report, I will print all three chart together.
# To print each chart individuaLly, comment out the other two out of three statements below.
sns.distplot((q4_df['food_consumption_2005']-q4_df['food_consumption_2005'].mean())/q4_df['food_consumption_2005'].std())
sns.distplot((q4_df['gdp_2005']-q4_df['gdp_2005'].mean())/q4_df['gdp_2005'].std())
sns.distplot((q4_df['life_expectancy_2005']-q4_df['life_expectancy_2005'].mean())/q4_df['life_expectancy_2005'].std());
# Print list of countries where the life expectancies are low.
q4_df['life_expectancy_2005'].loc[q4_df['life_expectancy_2005'] < \
                                  (q4_df['life_expectancy_2005'].mean()-q4_df['life_expectancy_2005'].std())]
q4_df.corr()
# Visualising the data and correlation using scatter plot. 
# The size of the bubbles are static and not influenced by any parameter. 
N = len(q4_df)
x = q4_df['food_consumption_2005']
y = q4_df['life_expectancy_2005']
colors = np.random.rand(N)
area = np.pi * (q4_df['gdp_2005'])**0.3  # This is to scale the size of the bubbles based on corresponding countries' GDP.

plt.scatter(x, y, s=area, c=colors, alpha=0.5, label='Countries GDP')
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
plt.xlabel('Food Consumption (kcal/person/day)')           # and here ?
plt.ylabel('Life Expectancies (years)')          # and here ?
plt.title('How Food Consumption Affect Life Expectancies')   # and here ?
plt.legend()
plt.grid(True)
plt.show()
# Select countries with extreme food consumption
q4_df_extreme_high_food_consumption = q4_df.loc[q4_df['food_consumption_2005'] > 3700]
q4_df_extreme_high_food_consumption
q4_df_extreme_high_food_consumption.corr()
# Create a DataFrame with all four indicators (Work Hours is excluded)
q5_df = pd.DataFrame(dict(gdp_2005 = gdp_2005, food_consumption_2005 = food_consumption_2005, \
                           alcohol_consumption_2005 = alcohol_consumption_2005, \
                           life_expectancy_2005 = life_expectancy_2005)).dropna()

q5_df.describe()
# Plot normalised value (using Z-score normalization) of columns.
# For this report, I will print all the chart together.
# To print each chart individuaLly, comment out the other three out of four statements below.
sns.distplot((q5_df['food_consumption_2005']-q5_df['food_consumption_2005'].mean())/q5_df['food_consumption_2005'].std())
sns.distplot((q5_df['gdp_2005']-q5_df['gdp_2005'].mean())/q5_df['gdp_2005'].std())
sns.distplot((q5_df['alcohol_consumption_2005']-q5_df['alcohol_consumption_2005'].mean())/q5_df['alcohol_consumption_2005'].std())
sns.distplot((q5_df['life_expectancy_2005']-q5_df['life_expectancy_2005'].mean())/q5_df['life_expectancy_2005'].std());
q5_df.corr()
# Isolating dataset, only taking low GDP countries with high alcohol consumption.
q5_df_gdp_below_mean = q5_df.loc[q5_df['gdp_2005'] < q5_df['gdp_2005'].mean()]
q5_df_gdp_below_mean_and_alc_above_mean = q5_df_gdp_below_mean.loc[q5_df['alcohol_consumption_2005'] \
                                                                   > q5_df_gdp_below_mean['alcohol_consumption_2005'].mean()]
q5_df_gdp_below_mean_and_alc_above_mean.describe()
q5_df_gdp_below_mean_and_alc_above_mean.corr()
N = len(q5_df_gdp_below_mean_and_alc_above_mean)
x = q5_df_gdp_below_mean_and_alc_above_mean['alcohol_consumption_2005']
y = q5_df_gdp_below_mean_and_alc_above_mean['life_expectancy_2005']
colors = np.random.rand(N)
area = np.pi * (q5_df_gdp_below_mean_and_alc_above_mean['gdp_2005'])**0.3

plt.scatter(x, y, s=area, c=colors, alpha=0.5, label='Countries GDP')
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
plt.xlabel('Alcohol Consumption (litres/person)') 
plt.ylabel('Life Expectancies (years)') 
plt.title('How Alcohol Consumption Affect Life Expectancies in low GDP countries') 
plt.legend()
plt.grid(True)
plt.show()
# Isolating dataset to include only countries with moderate food consumption as well.
q5_df_gdp_below_mean_and_alc_above_mean_foodabovemean = \
q5_df_gdp_below_mean_and_alc_above_mean.loc[q5_df_gdp_below_mean_and_alc_above_mean['food_consumption_2005'] \
                                            > (q5_df_gdp_below_mean_and_alc_above_mean['food_consumption_2005'].mean()\
                                               - q5_df_gdp_below_mean_and_alc_above_mean['food_consumption_2005'].std())]
q5_df_gdp_below_mean_and_alc_above_mean_foodmoderate = \
q5_df_gdp_below_mean_and_alc_above_mean_foodabovemean.loc[q5_df_gdp_below_mean_and_alc_above_mean['food_consumption_2005'] \
                                            < (q5_df_gdp_below_mean_and_alc_above_mean['food_consumption_2005'].mean()\
                                               + q5_df_gdp_below_mean_and_alc_above_mean['food_consumption_2005'].std())]
q5_df_gdp_below_mean_and_alc_above_mean_foodmoderate.describe()
q5_df_gdp_below_mean_and_alc_above_mean_foodmoderate.corr()
# Visualising the data and correlation using scatter plot. 
# The size of the bubbles are static and not influenced by any parameter. 
N = len(q5_df_gdp_below_mean_and_alc_above_mean_foodmoderate)
x = q5_df_gdp_below_mean_and_alc_above_mean_foodmoderate['alcohol_consumption_2005']
y = q5_df_gdp_below_mean_and_alc_above_mean_foodmoderate['life_expectancy_2005']
colors = np.random.rand(N)

# This is to scale the size of the bubbles based on corresponding countries' GDP.
area = np.pi * (q5_df_gdp_below_mean_and_alc_above_mean_foodmoderate['gdp_2005'])**0.3 

plt.scatter(x, y, s=area, c=colors, alpha=0.5, label='Countries GDP')
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
plt.xlabel('Alcohol Consumption (litres/person)') 
plt.ylabel('Life Expectancies (years)') 
plt.title('How Alcohol Consumption Affect Life Expectancies in low GDP countries') 
plt.legend()
plt.grid(True)
plt.show()