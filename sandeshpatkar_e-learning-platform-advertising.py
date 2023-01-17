import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('../input/fcc.txt')

df.head()
df.shape
import seaborn as sns
def create_df(series, norm):

    if norm == 1:

        val = series.value_counts(normalize = True)*100

        df = pd.DataFrame(val)

        df['index'] = df.index

    elif norm == 0:

        val = series.value_counts()

        df = pd.DataFrame(val)

        df['index'] = df.index

    return df
job_interest = create_df(df.JobRoleInterest, 1)
job_interest.head()
interested = df['JobRoleInterest'].dropna()

split_jobs = interested.str.split(',')
no_interests = split_jobs.apply(lambda x: len(x))

no_interests.value_counts(normalize = True)*100
def plot_bar(x,y,data, title):

    sns.barplot(x = x, y = y, data = data).set_title(title)

    plt.xticks(rotation = 90)
plot_bar(x = 'index', y = 'JobRoleInterest', data = job_interest[:10], title = 'Job Role Interests')
web_n_mobile = interested.str.contains('Web Developer|Mobile Developer')

exact = create_df(web_n_mobile, 1)
exact.head()
#ax = plot_bar(x = 'index', y = 'JobRoleInterest', data = exact, title = 'Participants interested in Web/Mobile Development')

sns.barplot(x = 'index', y = 'JobRoleInterest', data = exact)

plt.xlabel('Job Roles')

plt.ylabel('Percentage')

plt.xticks([0,1],['Other', 'Web/Mobile Development'])
df.JobRoleInterest.isnull().sum()
dfjobs = df.dropna(subset = ['JobRoleInterest']).copy()

dfjobs.shape
dfjobs.CountryLive.isnull().sum()
country = create_df(dfjobs.CountryLive, 1)

country.head()
plot_bar(x = 'index', y = 'CountryLive', data = country.head(5), title = 'Country wise distribution: Normalized')
country_normalize = create_df(dfjobs.CountryLive, 0)

country_normalize.head()
plot_bar('index', 'CountryLive', country_normalize.head(), title = 'Country wise distribution')
dfjobs[dfjobs['MonthsProgramming'] == 0].shape
dfjobs['MonthsProgramming'].replace(0, 1, inplace = True)
dfjobs[dfjobs['MonthsProgramming'] == 0].shape
dfjobs['moneymonth'] = dfjobs['MoneyForLearning']/dfjobs['MonthsProgramming']

dfjobs.moneymonth.head()
dfjobs.moneymonth.isnull().sum()
#removing the null values in moneymonth

dfjobs = dfjobs.dropna(subset = ['moneymonth'])

dfjobs.moneymonth.isnull().sum()
#removing the null values in CountryLive

dfjobs = dfjobs.dropna(subset = ['CountryLive'])

dfjobs.CountryLive.isnull().sum()

dfjobs.shape
dfjobs_mean = dfjobs.groupby(['CountryLive']).mean()
dfjobs4 = dfjobs_mean['moneymonth'][['United States of America', 'India', 'United Kingdom', 'Canada']]

dfjobs4.head()
dfjobs.head()
top_countries = dfjobs[(dfjobs['CountryLive'] == 'United States of America') |(dfjobs['CountryLive'] == 'India') | (dfjobs['CountryLive'] =='United Kingdom') |(dfjobs['CountryLive'] =='Canada')]
top_countries.CountryLive.unique()
sns.boxplot(x = 'CountryLive', y = 'moneymonth', data = top_countries)

plt.title('Country-wise monthly spend')

plt.xlabel('Country')

plt.ylabel('Money per month (US Dollars)')

plt.xticks(range(4),['US', 'UK', 'India', 'Cananda'])
less_than10k = top_countries[top_countries['moneymonth'] <= 10000]
sns.boxplot(x = 'CountryLive', y = 'moneymonth', data = less_than10k)

plt.title('Country-wise monthly spend')

plt.xlabel('Country')

plt.ylabel('Money per month (US Dollars)')

plt.xticks(range(4),['US', 'UK', 'India', 'Cananda'])
def money_check(money, country):

    x = less_than10k[(less_than10k['moneymonth'] >= money) & (less_than10k['CountryLive'] == country)]

    return x
india_outliers = money_check(2000, 'India')

india_outliers
less_than10k = less_than10k.drop(india_outliers.index)
us_outliers = money_check(5000, 'United States of America')

us_outliers
money_check(5000, 'United Kingdom')
money_check(5000, 'Canada')
less_than_3months = less_than10k[(less_than10k['MonthsProgramming'] <= 3) & (less_than10k['moneymonth'] > 5000)]

less_than_3months
less_than10k = less_than10k.drop(less_than_3months.index)

less_than10k
less_than10k.groupby('CountryLive').mean()['moneymonth']
less_than10k['CountryLive'].value_counts()