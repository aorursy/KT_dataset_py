import pandas as pd

from datetime import datetime as dt

import re

import numpy as np
jobs = pd.read_csv('/kaggle/input/data-jobs-listings-glassdoor/glassdoor.csv', parse_dates=['job.discoverDate'])

jobs.head(5)
jobs.shape
jobs.columns
best_columns = jobs[['gaTrackerData.empName', 'gaTrackerData.jobTitle', 'map.country', 'gaTrackerData.location', 'header.sponsored', 'job.description', 'job.discoverDate', 'salary.salaries', 'overview.industry', 'overview.sector', 'overview.revenue', 'salary.country.currency.currencyCode']]



del jobs



best_columns.head(5)
best_columns = best_columns.dropna(subset=['map.country'])

#best_columns.head(10)
countries = pd.read_csv('/kaggle/input/data-jobs-listings-glassdoor/country_names_2_digit_codes.csv')

countries.head(5)
## Transforming the countries' name



dict_countries = countries.to_dict('records')

for country in dict_countries:

  best_columns = best_columns.replace(to_replace={'map.country':country['Code']}, value={'map.country':country['Name']})



best_columns.head(10)
# Getting the countries that I want

countries_iwant = ['Italy', 'United States', 'Austria', 'Canada', 'Portugal', 'United Kingdom', 'Switzerland', 'Estonia', 'Germany', 'Netherlands', 'Ireland', 'Spain']
main_df = best_columns[best_columns['map.country'].isin(countries_iwant)]



del best_columns

del countries



main_df.head(10)
main_df['day'] = main_df['job.discoverDate'].dt.day

main_df['month'] = main_df['job.discoverDate'].dt.month

main_df = main_df.drop(['job.discoverDate'], axis='columns')

main_df.head(5)
main_df.shape
# Function to search and select only the data scientist jobs

# With this function you can look for any job that you want, using RegEx.



def search_job(job_title):

  re_expression = '(?<=data.)(scientist)'

  job_title = str(job_title)

  job_title = job_title.lower()

  is_the_job = re.search(re_expression, job_title)

  if is_the_job:

    return True

  else:

    return False
main_df = main_df[main_df['gaTrackerData.jobTitle'].apply(search_job)==True]

main_df.head(10)
main_df.shape
main_df.dtypes
import seaborn as sb

import matplotlib.pyplot as plt



sb.set_style('dark')

sb.set_context('paper')
# About the countries

plt.figure(figsize=(20,10))

sb.countplot(x='map.country', data=main_df)
# About the job's sectors

plt.figure(figsize=(20,10))

count_sector = sb.countplot(x='overview.sector', data=main_df, palette='muted')

for tick in count_sector.get_xticklabels():

  tick.set_rotation(55)

count_sector.set_xlabel('Comapany Sector')
# Working with the time variable

plt.figure(figsize=(20,10))

sb.countplot(y='month' , data=main_df, hue='map.country')
# How many currencies are described the salaries?



main_df['salary.country.currency.currencyCode'].unique()
s_salaries = pd.read_csv('/kaggle/input/data-jobs-listings-glassdoor/glassdoor_salary_salaries.csv')

s_salaries.head(10)
new_main_df = pd.merge(main_df, s_salaries, how='left', left_on='salary.salaries', right_on='id')



new_main_df = new_main_df.dropna(subset=['salary.salaries'])



new_main_df.head(10)
new_main_df['salary.salaries.val.payPeriod'].unique()
annual_s = new_main_df[new_main_df['salary.salaries.val.payPeriod'] == 'ANNUAL']



monthly_s = new_main_df[new_main_df['salary.salaries.val.payPeriod'] == 'MONTHLY']



hourly_s = new_main_df[new_main_df['salary.salaries.val.payPeriod'] == 'HOURLY']



print('Annual dataset size: ', annual_s.shape)

print('Monthly dataset size: ', monthly_s.shape)

print('Hourly dataset size: ', hourly_s.shape)
f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 8), sharex=True)



sb.boxplot(x='salary.salaries.val.salaryPercentileMap.payPercentile10', y='map.country', data=annual_s, palette='vlag', orient='h', ax=ax1)

sb.swarmplot(x='salary.salaries.val.salaryPercentileMap.payPercentile10', y='map.country', data=annual_s, size=2, color=".3", linewidth=0, ax=ax1)



sb.boxplot(x='salary.salaries.val.salaryPercentileMap.payPercentile50', y='map.country', data=annual_s, palette='vlag', orient='h', ax=ax2)

sb.swarmplot(x='salary.salaries.val.salaryPercentileMap.payPercentile50', y='map.country', data=annual_s, size=2, color=".3", linewidth=0, ax=ax2)



sb.boxplot(x='salary.salaries.val.salaryPercentileMap.payPercentile90', y='map.country', data=annual_s, palette='vlag', orient='h', ax=ax3)

sb.swarmplot(x='salary.salaries.val.salaryPercentileMap.payPercentile90', y='map.country', data=annual_s, size=2, color=".3", linewidth=0, ax=ax3)
# Removing outliers



an_10 = annual_s['salary.salaries.val.salaryPercentileMap.payPercentile10']

annual_10 = an_10.between(an_10.quantile(.05), an_10.quantile(.95))



# Here I'm getting the index rows where the value it's True and True means outlier

index_names_10 = annual_s[~annual_10].index



# Droping the values where index it's True

annual_p10_no_outliers = annual_s.drop(index_names_10, axis=0, inplace=False)
an_50 = annual_s['salary.salaries.val.salaryPercentileMap.payPercentile50']

annual_50 = an_50.between(an_50.quantile(.05), an_50.quantile(.95))



index_names_50 = annual_s[~annual_50].index



annual_p50_no_outliers = annual_s.drop(index_names_50, axis=0, inplace=False)
an_90 = annual_s['salary.salaries.val.salaryPercentileMap.payPercentile90']

annual_90 = an_90.between(an_90.quantile(.05), an_90.quantile(.95))



index_names_90 = annual_s[~annual_90].index



annual_p90_no_outliers = annual_s.drop(index_names_90, axis=0, inplace=False)
f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 8), sharex=True)



sb.boxplot(x='salary.salaries.val.salaryPercentileMap.payPercentile10', y='map.country', data=annual_p10_no_outliers, palette='vlag', orient='h', ax=ax1)

sb.swarmplot(x='salary.salaries.val.salaryPercentileMap.payPercentile10', y='map.country', data=annual_p10_no_outliers, size=2, color=".3", linewidth=0, ax=ax1)



sb.boxplot(x='salary.salaries.val.salaryPercentileMap.payPercentile50', y='map.country', data=annual_p50_no_outliers, palette='vlag', orient='h', ax=ax2)

sb.swarmplot(x='salary.salaries.val.salaryPercentileMap.payPercentile50', y='map.country', data=annual_p50_no_outliers, size=2, color=".3", linewidth=0, ax=ax2)



sb.boxplot(x='salary.salaries.val.salaryPercentileMap.payPercentile90', y='map.country', data=annual_p90_no_outliers, palette='vlag', orient='h', ax=ax3)

sb.swarmplot(x='salary.salaries.val.salaryPercentileMap.payPercentile90', y='map.country', data=annual_p90_no_outliers, size=2, color=".3", linewidth=0, ax=ax3)
f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 8), sharex=True)



sb.boxplot(x='salary.salaries.val.salaryPercentileMap.payPercentile10', y='map.country', data=monthly_s, palette='vlag', orient='h', ax=ax1)

sb.swarmplot(x='salary.salaries.val.salaryPercentileMap.payPercentile10', y='map.country', data=monthly_s, size=2, color=".3", linewidth=0, ax=ax1)



sb.boxplot(x='salary.salaries.val.salaryPercentileMap.payPercentile50', y='map.country', data=monthly_s, palette='vlag', orient='h', ax=ax2)

sb.swarmplot(x='salary.salaries.val.salaryPercentileMap.payPercentile50', y='map.country', data=monthly_s, size=2, color=".3", linewidth=0, ax=ax2)



sb.boxplot(x='salary.salaries.val.salaryPercentileMap.payPercentile90', y='map.country', data=monthly_s, palette='vlag', orient='h', ax=ax3)

sb.swarmplot(x='salary.salaries.val.salaryPercentileMap.payPercentile90', y='map.country', data=monthly_s, size=2, color=".3", linewidth=0, ax=ax3)
f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 8), sharex=True)



sb.boxplot(x='salary.salaries.val.salaryPercentileMap.payPercentile10', y='map.country', data=hourly_s, palette='vlag', orient='h', ax=ax1)

sb.swarmplot(x='salary.salaries.val.salaryPercentileMap.payPercentile10', y='map.country', data=hourly_s, size=2, color=".3", linewidth=0, ax=ax1)



sb.boxplot(x='salary.salaries.val.salaryPercentileMap.payPercentile50', y='map.country', data=hourly_s, palette='vlag', orient='h', ax=ax2)

sb.swarmplot(x='salary.salaries.val.salaryPercentileMap.payPercentile50', y='map.country', data=hourly_s, size=2, color=".3", linewidth=0, ax=ax2)



sb.boxplot(x='salary.salaries.val.salaryPercentileMap.payPercentile90', y='map.country', data=hourly_s, palette='vlag', orient='h', ax=ax3)

sb.swarmplot(x='salary.salaries.val.salaryPercentileMap.payPercentile90', y='map.country', data=hourly_s, size=2, color=".3", linewidth=0, ax=ax3)
!pip install langdetect

from langdetect import detect



!pip install unidecode

import unidecode
def replace_html(text):

  re_expression = '(<\w+>|</\w+>|\\n|<\w+/>|&quot;|&egrave|http\S+|[-|0-9]|\*|,|;|&)'

  clean_text = re.sub(re_expression, ' ', text)

  ready_text = unidecode.unidecode(clean_text.lower())

  return ready_text
new_main_df['job.description'] = new_main_df['job.description'].map(replace_html)

new_main_df.head(5)
# Separating the jobs by language



new_main_df['language'] = new_main_df['job.description'].apply(detect)

new_main_df.head(10)
new_main_df['language'].unique()
italian_df = new_main_df[new_main_df['language'].isin(['it'])]

english_df = new_main_df[new_main_df['language'].isin(['en'])]

german_df = new_main_df[new_main_df['language'].isin(['de'])]

spanish_df = new_main_df[new_main_df['language'].isin(['es'])]

netherlands_df = new_main_df[new_main_df['language'].isin(['nl'])]

france_df = new_main_df[new_main_df['language'].isin(['fr'])]

portuguese_df = new_main_df[new_main_df['language'].isin(['pt'])]
# Dataset size by language



print('Italian dataset size: ', italian_df.shape)

print('English dataset size: ', english_df.shape)

print('German dataset size: ', german_df.shape)

print('Spanish dataset size: ', spanish_df.shape)

print('Portuguese dataset size: ', portuguese_df.shape)
# Function to put together all descriptions



def aggregate_strings(dataframe_series):

  big_string = ''

  for text in dataframe_series:

    big_string += text + ' '

  return big_string
# Putting together all descriptions by language



english_descs = aggregate_strings(english_df['job.description'])

italian_descs = aggregate_strings(italian_df['job.description'])

german_descs = aggregate_strings(german_df['job.description'])

spanish_descs = aggregate_strings(spanish_df['job.description'])

portuguese_descs = aggregate_strings(portuguese_df['job.description'])
# Packages to figure out the most used words on jobs descriptions



import nltk

nltk.download('stopwords')

nltk.download('punkt')

from string import punctuation

from wordcloud import WordCloud
# Portuguese WordCloud



portuguese_stopwords = set(nltk.corpus.stopwords.words('portuguese') + list(punctuation))

portuguese_wordcloud = WordCloud(max_words=100, stopwords=portuguese_stopwords, width=800, height=400).generate(portuguese_descs)



plt.figure(figsize=(20,10))

plt.imshow(portuguese_wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
# English WordCloud



english_stopwords = set(nltk.corpus.stopwords.words('english') + list(punctuation))

english_wordcloud = WordCloud(max_words=100, stopwords=english_stopwords, width=800, height=400).generate(english_descs)



plt.figure(figsize=(20,10))

plt.imshow(english_wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
# Italian WordCloud



italian_stopwords = set(nltk.corpus.stopwords.words('italian') + list(punctuation))

italian_wordcloud = WordCloud(max_words=100, stopwords=italian_stopwords, width=800, height=400).generate(italian_descs)



plt.figure(figsize=(20,10))

plt.imshow(italian_wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
# Spanish WordCloud



spanish_stopwords = set(nltk.corpus.stopwords.words('spanish') + list(punctuation))

spanish_wordcloud = WordCloud(max_words=100, stopwords=spanish_stopwords, width=800, height=400).generate(spanish_descs)



plt.figure(figsize=(20,10))

plt.imshow(spanish_wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
# German WordCloud



german_stopwords = set(nltk.corpus.stopwords.words('german') + list(punctuation))

german_wordcloud = WordCloud(max_words=100, stopwords=german_stopwords, width=800, height=400).generate(german_descs)



plt.figure(figsize=(20,10))

plt.imshow(german_wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()