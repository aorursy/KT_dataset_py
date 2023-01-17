import pandas as pd

import matplotlib.pyplot as plt 

%matplotlib inline

pd.set_option('display.max_columns', 200)
survey_dataset = pd.read_csv('/kaggle/input/2017-fCC-New-Coders-Survey-Data.csv', low_memory = 0)
survey_dataset.shape
survey_dataset.head()
survey_dataset.columns
survey_dataset['JobRoleInterest'].head()
survey_dataset['JobRoleInterest'].value_counts(normalize = True, dropna = False) * 100
job_interest_clean = survey_dataset['JobRoleInterest'].dropna()
job_interest_clean.isnull().sum()
job_interest_clean.head()
job_interest_clean = job_interest_clean.str.split(',')
job_interest_clean[0:5]
job_interest_clean = job_interest_clean.apply(lambda x : len(x))
job_interest_clean[0:3]
job_interest_fd = job_interest_clean.value_counts(normalize = True)*100

job_interest_fd
job_interest_fd.plot.bar() 
plt.style.available
plt.style.use('fivethirtyeight')
job_interest_fd.plot.bar()
job_interest_fd.plot.bar(figsize = (8,8))

plt.xticks(rotation = 0)

plt.title('Number of subjects, respondents interested in learning', y = 1.05, size = 13)

plt.xlabel('Number of Subjects respondents interested in', size = 11)

plt.ylabel('Percentage', size = 11)

plt.tick_params(axis = 'both', which = 'major', labelsize = 13)

plt.axhline(0, color = 'black')



jobrole_interest = survey_dataset[survey_dataset['JobRoleInterest'].notnull()]
countries_absolute_frequencies = jobrole_interest['CountryLive'].value_counts()

countries_relative_frequencies = jobrole_interest['CountryLive'].value_counts(normalize = True)*100
pd.DataFrame(data = {'Absolute Frequency' : countries_absolute_frequencies, 'Relative Frequency' : countries_relative_frequencies })
survey_dataset['MonthsProgramming'].value_counts()
job_role_copy = jobrole_interest.copy()
job_role_copy['MonthsProgramming'].replace(0,1,inplace=True)
job_role_copy['Monthly_Spend'] = job_role_copy['MoneyForLearning'] / job_role_copy['MonthsProgramming']
job_role_copy['Monthly_Spend'].isnull().sum()
job_role_copy = job_role_copy[job_role_copy['Monthly_Spend'].notnull()]
job_role_copy['CountryLive'].isnull().sum()
job_role_copy = job_role_copy[job_role_copy['CountryLive'].notnull()]
job_role_copy['CountryLive'].isnull().sum()
countries_grouped = job_role_copy.groupby('CountryLive').mean() 
countries_grouped.head() 
countries_grouped['Monthly_Spend'][["United States of America", "India", "United Kingdom", "Canada"]]
import seaborn as sns
four_countries = job_role_copy[job_role_copy['CountryLive'].str.contains('United States of America|India|United Kingdom|Canada')]
plt.figure(figsize = (10,10))

sns.boxplot(x = 'CountryLive', y = 'Monthly_Spend', data = four_countries)

sns.set(style = 'whitegrid')

plt.title('Money Spend Per Month Based on Country', size = 13)

plt.xlabel('Country', size = 13)

plt.ylabel('Monthly Spend', size = 13)

plt.yticks(range(0,100000,10000))

plt.show()

job_role_copy = job_role_copy[job_role_copy['Monthly_Spend'] < 20000]
countries_grouped_mod = job_role_copy.groupby('CountryLive').mean()
countries_grouped_mod['Monthly_Spend'][["United States of America", "India", "United Kingdom", "Canada"]]
four_countries_mod = job_role_copy[job_role_copy['CountryLive'].str.contains('United States of America|India|United Kingdom|Canada')]
plt.figure(figsize = (10,10))

sns.boxplot(x = 'CountryLive', y = 'Monthly_Spend', data = four_countries_mod)

sns.set(style = 'whitegrid')

plt.title('Money Spend Per Month Based on Country', size = 13)

plt.xlabel('Country', size = 13)

plt.ylabel('Monthly Spend', size = 13)

plt.show()
india_data = four_countries_mod[(four_countries_mod['CountryLive'] == 'India') & (four_countries_mod['Monthly_Spend'] >= 2500)]

india_data
four_countries_mod = four_countries_mod.drop(india_data.index)
usa_data = four_countries_mod[(four_countries_mod['CountryLive'] == 'United States of America') & (four_countries_mod['Monthly_Spend'] >= 6000)]
usa_data
no_bootcamp = four_countries_mod[(four_countries_mod['CountryLive'] == 'United States of America') & (four_countries_mod['Monthly_Spend'] >= 6000) & (four_countries_mod['AttendedBootcamp'] == 0)]
no_bootcamp
four_countries_mod = four_countries_mod.drop(no_bootcamp.index)
less_months = four_countries_mod[(four_countries_mod['CountryLive'] == 'United States of America') & (four_countries_mod['Monthly_Spend'] >= 6000) & (four_countries_mod['MonthsProgramming'] <= 3)]
less_months['MonthsProgramming']
four_countries_mod = four_countries_mod.drop(less_months.index)
canada_outliers = four_countries_mod[(four_countries_mod['CountryLive'] == 'Canada') & (four_countries_mod['Monthly_Spend'] > 4000)]
canada_outliers
four_countries_mod = four_countries_mod.drop(canada_outliers.index)
four_countries_mod.groupby('CountryLive').mean()['Monthly_Spend']
four_countries_mod['CountryLive'].value_counts(normalize = True)*100