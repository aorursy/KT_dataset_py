# Loading necessary libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# loading data
url = '../input/ks-projects-201801.csv'
with open(url , 'r') as f:
    df = pd.read_csv(f).fillna(0)
df.info()
# get mean of projects in all categories
mean_categories = df.pivot_table('pledged', index='category', columns='state', aggfunc='mean').sort_values('successful', ascending=False)
# take only successful and format output -> float to percent
profitable_categories = mean_categories['successful']
print(profitable_categories[0:10].apply(lambda x: '{0:.0f}'.format(x)))
# count projects in all categories
count_categories = df.pivot_table('pledged', index='category', columns='state', aggfunc='count')
# get percent of each state and sort by succesful
percent_count_successful = count_categories.apply(lambda x: x*100/x.sum(), axis=1).sort_values('successful', ascending=False)
# take only successful and format output -> float to percent
successfull_categories = percent_count_successful['successful']
print(successfull_categories[0:10].apply(lambda x: '{0:.0f}%'.format(x)))
merged_data = pd.concat([profitable_categories, successfull_categories], axis=1, join='inner')
merged_data.columns = ['pledged','percent']
percent_to_values = merged_data.apply(lambda x: x['pledged']*x['percent']/100, axis=1)
result_table = pd.concat([merged_data['pledged'].to_frame(), percent_to_values], axis=1, join='inner')
result_table.columns = ['pledged','successfull']
ax = result_table[0:15].sort_values('successfull').plot(kind='barh', figsize=(15, 7), legend=True, width=0.75)
ax.set_title('Top categories with best profit and high percent of success')
ax.set_xlabel("USD")
ax.legend(['Pledged(USD)', '% success'], loc=0)
df['launched'] = pd.to_datetime(df['launched'])
df['month'] = df['launched'].dt.month
successfull_months = df[df['state'] == 'successful']['month'].value_counts().rename('successful starts').sort_index()
unsuccessfull_months = df[df['state'] != 'successful']['month'].value_counts().rename('unsuccessful starts').sort_index()
most_successful_months = pd.concat([successfull_months, unsuccessfull_months], axis=1, join='inner')
most_successful_months['backers activity (scaled)'] = df.groupby('month').aggregate(lambda x:x.sum()/200)['backers']
most_successful_months.plot(figsize=(15, 5), legend=True, title='Most successful months')