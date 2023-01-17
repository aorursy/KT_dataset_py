import numpy as np # linear algebra
from numpy import log10, ceil, ones
from numpy.linalg import inv 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # prettier graphs
import matplotlib.pyplot as plt # need dis too
%matplotlib inline 
from IPython.display import HTML # for da youtube memes
import itertools # let's me iterate stuff
from datetime import datetime # to work with dates
import geopandas as gpd
from fuzzywuzzy import process
from shapely.geometry import Point, Polygon
import shapely.speedups
shapely.speedups.enable()
import fiona 
from time import gmtime, strftime
from shapely.ops import cascaded_union
import gc

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

sns.set_style('darkgrid') # looks cool, man
import os

df_cntry_codes = pd.read_csv("../input/countries-iso-codes/wikipedia-iso-country-codes.csv")
df_cntry_codes = df_cntry_codes.rename(index=str, columns={'Alpha-2 code': 'country_code', 'English short name lower case' : 'country'})
df_lenders = pd.read_csv("../input/additional-kiva-snapshot/lenders.csv")
df_lenders.head()
df_lenders[df_lenders['permanent_name'] == 'mikedev10']
df_loans_lenders = pd.read_csv("../input/additional-kiva-snapshot/loans_lenders.csv")
df_loans_lenders.head()
df_loans = pd.read_csv("../input/additional-kiva-snapshot/loans.csv")
df_loans['posted_time'] = pd.to_datetime(df_loans['posted_time'], format='%Y-%m-%d %H:%M:%S').dt.round(freq='D')
df_loans['raised_time'] = pd.to_datetime(df_loans['posted_time'], format='%Y-%m-%d %H:%M:%S').dt.round(freq='D')
df_loans['posted_month'] = df_loans['posted_time'].dt.to_period('M').dt.to_timestamp()
df_loans.head()
# thanks to
# https://stackoverflow.com/questions/38651008/splitting-multiple-columns-into-rows-in-pandas-dataframe

# STEP 1 - explode lenders out of of column entry into multiple rows
def explode(df, columns):
    idx = np.repeat(df.index, df[columns[0]].str.len())
    a = df.T.reindex_axis(columns).values
    concat = np.concatenate([np.concatenate(a[i]) for i in range(a.shape[0])])
    p = pd.DataFrame(concat.reshape(a.shape[0], -1).T, idx, columns)
    return pd.concat([df.drop(columns, axis=1), p], axis=1).reset_index(drop=True)

# THE BELOW WAS REPLACED WITH OUTPUT AND READ OF A FILE TO GET AROUND CONSTANT NOTEBOOK CRASHING AND ELIMINATE DUPLICATES
#df_exp = df_loans_lenders
#df_exp['lenders'] = df_exp['lenders'].str.split(',')
#df_exp = explode(df_exp, ['lenders'])
#df_exp = df_exp.rename(index=str, columns={'lenders': 'permanent_name'})
#df_exp['permanent_name'] = df_exp['permanent_name'].str.strip()
#df_exp = df_exp.drop_duplicates()
df_exp = pd.read_csv("../input/kiva-lender-helper/df_exp.csv")
#dupe check
#df_exp[df_exp['loan_id'] == 885412]
# STEP 2 - map users to countries
df_lender_cntry = df_exp.merge(df_lenders[['permanent_name', 'country_code']], on='permanent_name')
df_lender_cntry.dropna(axis=0, how='any', inplace=True)
#df_lender_cntry

# STEP 3 - merge users to loans and aggregate count by country and day
df_cntry_cnts = df_lender_cntry.merge(df_loans[['loan_id', 'posted_time']], on='loan_id')[['country_code', 'posted_time']].groupby(['country_code', 'posted_time']).size().reset_index(name='counts')
#df_cntry_cnts.head()

# STEP 4 - let's make life easier with these country codes...
df_cntry_cnts = df_cntry_cnts.merge(df_cntry_codes[['country_code', 'country']], on='country_code', how='left')
df_cntry_cnts['country'] = np.where(df_cntry_cnts['country_code'] == 'SS', 'South Sudan', df_cntry_cnts['country'])  
df_cntry_cnts['country'] = np.where(df_cntry_cnts['country_code'] == 'XK', 'Kosovo', df_cntry_cnts['country'])  


df_cntry_cnts.head()
plt.figure(figsize=(15,8))
plotSeries = df_lenders['country_code'].value_counts()
ax = sns.barplot(plotSeries.head(30).values, plotSeries.head(30).index, color='c')
ax.set_title('Top 30 Lender Locations', fontsize=15)
ax.set(ylabel='Country (ISO-2 Abbrev)', xlabel='Lender Count')
plt.show()
plt.figure(figsize=(15,8))
plotSeries = df_lenders[df_lenders['country_code'] != 'US']['country_code'].value_counts()
ax = sns.barplot(plotSeries.head(29).values, plotSeries.head(29).index, color='b')
ax.set_title('Top 30 Lender Locations, Minus the US', fontsize=15)
ax.set(ylabel='Country (ISO-2 Abbrev)', xlabel='Lender Count')
plt.show()
df_cntry_sum = df_cntry_cnts.groupby(['country', 'country_code']).sum()
df_cntry_sum.reset_index(level=1, inplace=True)
df_cntry_sum.reset_index(level=0, inplace=True)
df_display = df_cntry_sum.sort_values('counts', ascending=False).head(30)

fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True)

sns.barplot(x='counts', y='country', data=df_display, color='c')

ax.set_title('Top 30 Locations by Lender Contribution Count', fontsize=15)
ax.set(ylabel='Country', xlabel='Number of Loan Contributions')
plt.show()
df_display = df_cntry_sum[df_cntry_sum['country_code'] != 'US'].sort_values('counts', ascending=False).head(30)

fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True)

sns.barplot(x='counts', y='country', data=df_display, color='b')

ax.set_title('Top 30 Locations by Lender Contribution Count, Minus the US', fontsize=15)
ax.set(ylabel='Country', xlabel='Number of Loan Contributions')
plt.show()
df_lender_sum = df_lenders.groupby(['country_code']).size().reset_index(name='counts_reg')
df_lender_sum = df_lender_sum.merge(df_cntry_sum, on='country_code')
df_lender_sum['loans_per_lender'] = df_lender_sum['counts'] / df_lender_sum['counts_reg']
df_lender_sum.head()
df_display = df_lender_sum.sort_values('loans_per_lender', ascending=False).head(30)

fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True)

sns.barplot(x='loans_per_lender', y='country', data=df_display, color='r')

ax.set_title('Top 30 Locations by Lender Contribution Count Per Registered User', fontsize=15)
ax.set(ylabel='Country', xlabel='Number of Loan Contributions')
plt.show()
df_display.head(10)
df_display = df_lender_sum[df_lender_sum['counts_reg'] >= 50].sort_values('loans_per_lender', ascending=False).head(30)

fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True)

sns.barplot(x='loans_per_lender', y='country', data=df_display, color='orange')

ax.set_title('Top 30 Locations by Lender Contribution Count Per Registered User, >= 50 Users', fontsize=15)
ax.set(ylabel='Country', xlabel='Number of Loan Contributions Per User')
plt.show()
df_lenders[df_lenders['country_code'] == 'OM'][['country_code', 'occupation', 'loan_because']].dropna(axis=0, how='any')
df_lenders[df_lenders['country_code'] == 'KZ'][['occupation', 'loan_because']].dropna(axis=0, how='all')
df_lenders[df_lenders['country_code'] == 'TH'][['occupation', 'loan_because']].dropna(axis=0, how='all')
df_lenders[df_lenders['country_code'] == 'CH'][['occupation', 'loan_because']].dropna(axis=0, how='all')
df_lenders[df_lenders['country_code'] == 'IR'][['occupation', 'loan_because']].dropna(axis=0, how='all')
df_display = df_lender_sum[df_lender_sum['counts_reg'] >= 1000].sort_values('loans_per_lender', ascending=False).head(30)

fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True)

sns.barplot(x='loans_per_lender', y='country', data=df_display, color='purple')

ax.set_title('Top 30 Locations by Lender Contribution Count Per Registered User, >= 1,000 Users', fontsize=15)
ax.set(ylabel='Country', xlabel='Number of Loan Contributions Per User')
plt.show()
df_display = df_lender_sum[df_lender_sum['counts_reg'] >= 10000].sort_values('loans_per_lender', ascending=False).head(30)

fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True)

sns.barplot(x='loans_per_lender', y='country', data=df_display, color='green')

ax.set_title('Top Locations by Lender Contribution Count Per Registered User, >= 10,000 Users', fontsize=15)
ax.set(ylabel='Country', xlabel='Number of Loan Contributions Per User')
plt.show()
# Youtube
HTML('Just look at the Netherlands go!  They are running a great campaign!<br><iframe width="560" height="315" src="https://www.youtube.com/embed/ELD2AwFN9Nc?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')
# clean up some data first
df_lenders['state'] = np.where(df_lenders['state'].str.len() <= 3, df_lenders['state'].str.upper().str.strip('.').str.strip(), df_lenders['state'].str.title())
df_lenders['state'] = np.where(df_lenders['state'].str.contains('California'), 'CA', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Texas'), 'TX', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('New York'), 'NY', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Florida'), 'FL', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Washington'), 'WA', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Illinois'), 'IL', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Colorado'), 'CO', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Pennsylvania'), 'PA', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Oregon'), 'OR', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Ohio'), 'OH', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Michigan'), 'MI', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Georgia'), 'GA', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Massachusetts'), 'MA', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Indiana'), 'IN', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Missouri'), 'MO', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Virginia'), 'VA', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Minnesota'), 'MN', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('North Carolina'), 'NC', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Arizona'), 'AZ', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Wisconsin'), 'WI', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Maryland'), 'MD', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('New Jersey'), 'NJ', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Kentucky'), 'KY', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Kansas'), 'KS', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Oklahoma'), 'OK', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Utah'), 'UT', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Tennessee'), 'TN', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('District of Columbia'), 'DC', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Iowa'), 'IA', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Connecticut'), 'CT', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Alabama'), 'AL', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Louisiana'), 'LA', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Idaho'), 'ID', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('South Carolina'), 'SC', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Maine'), 'ME', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Arkansas'), 'AR', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('New Mexico'), 'NM', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Hawaii'), 'HI', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Alaska'), 'AK', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('New Hampshire'), 'NH', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Nebraska'), 'NE', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Vermont'), 'VT', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Nevada'), 'NV', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Montana'), 'MT', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Rhode Island'), 'RI', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('West Virginia'), 'WV', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Mississippi'), 'MS', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Delaware'), 'DE', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('North Dakota'), 'ND', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Wyoming'), 'WY', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('South Dakota'), 'SD', df_lenders['state'])
df_lenders['state'] = np.where((df_lenders['state'].str.len() > 2) & (df_lenders['state'].str.len() <= 5), 
                               df_lenders['state'].str.upper().str.replace('.', '').str.strip(), df_lenders['state'])
plt.figure(figsize=(15,8))
plotSeries = df_lenders[df_lenders['country_code'] == 'US']['state'].value_counts()
ax = sns.barplot(plotSeries.head(30).values, plotSeries.head(30).index, color='c')
ax.set_title('Top 30 Lender State Locations', fontsize=15)
ax.set(ylabel='US State',
       xlabel='Lender Count')
plt.show()
# STEP 2 - map users to US states
df_lender_state = df_exp.merge(df_lenders[df_lenders['country_code'] == 'US'][['permanent_name', 'state']], on='permanent_name')
df_lender_state.dropna(axis=0, how='any', inplace=True)
#df_lender_cntry

# STEP 3 - merge users to loans and aggregate count by country and day
df_state_cnts = df_lender_state.merge(df_loans[['loan_id', 'posted_time']], on='loan_id')[['state', 'posted_time']].groupby(['state', 'posted_time']).size().reset_index(name='counts')

df_state_sum = df_state_cnts.groupby(['state']).sum()
df_state_sum.reset_index(level=0, inplace=True)

df_lender_sum_state = df_lenders[df_lenders['country_code'] == 'US'].groupby(['state']).size().reset_index(name='counts_reg')
df_lender_sum_state = df_lender_sum_state.merge(df_state_sum, on='state')
df_lender_sum_state['loans_per_lender'] = df_lender_sum_state['counts'] / df_lender_sum_state['counts_reg']

# let's merge in state populations here too
df_state_pop = pd.read_csv("../input/population-by-state/population.csv")
df_state_pop = df_state_pop.rename(index=str, columns={'State': 'state'})
df_lender_sum_state = df_lender_sum_state.merge(df_state_pop, on='state')
df_lender_sum_state['loans_per_capita'] = df_lender_sum_state['counts_reg'] / df_lender_sum_state['Population']

df_lender_sum_state.sort_values('counts_reg', ascending=False).head()
df_display = df_lender_sum_state.sort_values('loans_per_capita', ascending=False).head(30)

fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True)

sns.barplot(x='loans_per_capita', y='state', data=df_display, color='lightblue')

ax.set_title('Top 30 Lender State Locations, Loans Per Capita', fontsize=15)
ax.set(ylabel='US State', xlabel='Lender Count Per Capita')
plt.show()
df_display = df_lender_sum_state.sort_values('loans_per_lender', ascending=False).head(30)

fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True)

sns.barplot(x='loans_per_lender', y='state', data=df_display, color='salmon')

ax.set_title('Top 30 Lender State Locations, Loans Per User', fontsize=15)
ax.set(ylabel='US State', xlabel='Number of Loans Per User')
plt.show()
plt.figure(figsize=(12,6))
sns.distplot(df_lenders['loan_purchase_num'].fillna(0), bins=30)
plt.show()
df_lenders.sort_values('loan_purchase_num', ascending=False).head(20)[['permanent_name', 'city', 'state', 'country_code', 'occupation', 'loan_because', 'loan_purchase_num', 'num_invited']]
for x in range(0,10):
    print('99.' + str(x) + 'th percentile loan_purchase_num is: ' + str(df_lenders['loan_purchase_num'].quantile(0.99 + x/1000)))
df_lenders[df_lenders['loan_purchase_num'] >= 1000]['loan_purchase_num'].agg(['count', 'sum'])
for x in range(1,10):
    print(str(x * 10) + 'th percentile loan_purchase_num is: ' + str(df_lenders['loan_purchase_num'].quantile(x/10)))
for x in range(1,10):
    print('9' + str(x) + 'th percentile loan_purchase_num is: ' + str(df_lenders['loan_purchase_num'].quantile(.9 + x/100)))
fig, ax = plt.subplots(1, 1, figsize=(12,6))
sns.distplot(df_lenders[df_lenders['loan_purchase_num'] <= 50]['loan_purchase_num'].fillna(0), bins=30)

ax.set_title('Distribution - Number of Loan Contributions by User, <= 50 Loans', fontsize=15)
ax.set(xlabel='Loan Contributions by User')
plt.show()
fig, ax = plt.subplots(1, 1, figsize=(12,6))
sns.distplot(df_lenders[(df_lenders['loan_purchase_num'] <= 50) & (df_lenders['country_code'] == 'US')]['loan_purchase_num'].fillna(0), bins=30)

ax.set_title('Distribution - Number of Loan Contributions by User, <= 50 Loans, USA Only', fontsize=15)
ax.set(xlabel='Loan Contributions by User')
plt.show()
df_lenders.sort_values('num_invited', ascending=False).head(20)[['permanent_name', 'city', 'state', 'country_code', 'occupation', 'loan_because', 'loan_purchase_num', 'num_invited']]
df_loans['avg_funded'] = df_loans['funded_amount'] / df_loans['num_lenders_total']
df_loans.sort_values('avg_funded', ascending=False).head(20)[['loan_id', 'funded_amount', 'num_lenders_total', 'activity_name', 'description_translated']]
#df_whale = df_lenders.sort_values('loan_purchase_num', ascending=False).head(20)[['permanent_name']]
#remove outliers
outlier = 707
df_whale = df_lenders[df_lenders['loan_purchase_num'] >= outlier][['permanent_name']]
df_whale['whale'] = 'Y'
df_lenders = df_lenders.merge(df_whale, how='left', on='permanent_name')

# STEP 2 - map users to countries - exclude whales
df_lender_cntry = df_exp.merge(df_lenders[df_lenders['whale'].isnull()][['permanent_name', 'country_code']], on='permanent_name')
df_lender_cntry.dropna(axis=0, how='any', inplace=True)
#df_lender_cntry

# STEP 3 - merge users to loans and aggregate count by country and day
df_cntry_cnts = df_lender_cntry.merge(df_loans[['loan_id', 'posted_time']], on='loan_id')[['country_code', 'posted_time']].groupby(['country_code', 'posted_time']).size().reset_index(name='counts')
#df_cntry_cnts.head()

# STEP 4 - let's make life easier with these country codes...
df_cntry_cnts = df_cntry_cnts.merge(df_cntry_codes[['country_code', 'country']], on='country_code', how='left')
df_cntry_cnts['country'] = np.where(df_cntry_cnts['country_code'] == 'SS', 'South Sudan', df_cntry_cnts['country'])  
df_cntry_cnts['country'] = np.where(df_cntry_cnts['country_code'] == 'XK', 'Kosovo', df_cntry_cnts['country'])  

# sum up to country level
df_cntry_sum = df_cntry_cnts.groupby(['country', 'country_code']).sum()
df_cntry_sum.reset_index(level=1, inplace=True)
df_cntry_sum.reset_index(level=0, inplace=True)

# country divided by registered users...  slightly unfairly, counting whales as 0s now...
df_lender_sum = df_lenders.groupby(['country_code']).size().reset_index(name='counts_reg')
df_lender_sum = df_lender_sum.merge(df_cntry_sum, on='country_code')
df_lender_sum['loans_per_lender'] = df_lender_sum['counts'] / df_lender_sum['counts_reg']
df_display = df_lender_sum[df_lender_sum['counts_reg'] >= 1000].sort_values('loans_per_lender', ascending=False).head(30)

fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True)

sns.barplot(x='loans_per_lender', y='country', data=df_display, color='purple')

ax.set_title('Top 30 Locations by Lender Contribution Count Per Registered User, >= 1,000 Users, Minus Outliers', fontsize=15)
ax.set(ylabel='Country', xlabel='Number of Loan Contributions Per User')
plt.show()
df_display = df_lender_sum[df_lender_sum['counts_reg'] >= 10000].sort_values('loans_per_lender', ascending=False).head(30)

fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True)

sns.barplot(x='loans_per_lender', y='country', data=df_display, color='green')

ax.set_title('Top Locations by Lender Contribution Count Per Registered User, >= 10,000 Users, Minus Outliers', fontsize=15)
ax.set(ylabel='Country', xlabel='Number of Loan Contributions Per User')
plt.show()
# STEP 2 - map users to US states
df_lender_state = df_exp.merge(df_lenders[(df_lenders['country_code'] == 'US') & (df_lenders['whale'].isnull())][['permanent_name', 'state']], on='permanent_name')
df_lender_state.dropna(axis=0, how='any', inplace=True)
#df_lender_cntry

# STEP 3 - merge users to loans and aggregate count by country and day
df_state_cnts = df_lender_state.merge(df_loans[['loan_id', 'posted_time']], on='loan_id')[['state', 'posted_time']].groupby(['state', 'posted_time']).size().reset_index(name='counts')

df_state_sum = df_state_cnts.groupby(['state']).sum()
df_state_sum.reset_index(level=0, inplace=True)

df_lender_sum_state = df_lenders[df_lenders['country_code'] == 'US'].groupby(['state']).size().reset_index(name='counts_reg')
df_lender_sum_state = df_lender_sum_state.merge(df_state_sum, on='state')
df_lender_sum_state['loans_per_lender'] = df_lender_sum_state['counts'] / df_lender_sum_state['counts_reg']

# let's merge in state populations here too
df_state_pop = pd.read_csv("../input/population-by-state/population.csv")
df_state_pop = df_state_pop.rename(index=str, columns={'State': 'state'})
df_lender_sum_state = df_lender_sum_state.merge(df_state_pop, on='state')
df_lender_sum_state['loans_per_capita'] = df_lender_sum_state['counts_reg'] / df_lender_sum_state['Population']
# there's one guy in TT... adding in this registered user count to make that outlier disappear though.
df_display = df_lender_sum_state[df_lender_sum_state['counts_reg'] > 10].sort_values('loans_per_lender', ascending=False).head(30)

fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True)

sns.barplot(x='loans_per_lender', y='state', data=df_display, color='salmon')

ax.set_title('Top 30 Lender State Locations, Loans Per User', fontsize=15)
ax.set(ylabel='US State', xlabel='Number of Loans Per User')
plt.show()
scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]


data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = df_lender_sum_state['state'],
        z = df_lender_sum_state['loans_per_lender'],
        locationmode = 'USA-states',
        #text = df['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "loans per registered user")
        ) ]

layout = dict(
        title = 'Loan Contribution Count Per Registered User',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
py.iplot( fig, filename='d3-cloropleth-map' )
df_lenders['occupation'] = df_lenders['occupation'].str.title()
df_display = df_lenders['occupation'].value_counts().head(30).to_frame()

fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True)

sns.barplot(x='occupation', y=df_display.index, data=df_display, color='aqua')

ax.set_title('Top 30 Occupations - Registered Users', fontsize=15)
ax.set(ylabel='Occupation', xlabel='Number of Registered Lenders')
plt.show()

df_display = df_lenders.groupby('occupation')['loan_purchase_num'].sum().to_frame().sort_values('loan_purchase_num', ascending=False).head(30)

fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True)

sns.barplot(x='loan_purchase_num', y=df_display.index, data=df_display, color='darkcyan')

ax.set_title('Top 30 Occupations - Most Loan Contributions', fontsize=15)
ax.set(ylabel='Occupation', xlabel='Number of Loan Contributions')
plt.show()
df_occ_cnts = df_lenders['occupation'].value_counts().to_frame()
df_occ_cnts.reset_index(level=0, inplace=True)
df_occ_cnts = df_occ_cnts.rename(index=str, columns={'occupation': 'count_reg', 'index' : 'occupation'})

df_occ_loans = df_lenders.groupby('occupation')['loan_purchase_num'].sum().to_frame()
df_occ_loans.reset_index(level=0, inplace=True)
df_occ_loans = df_occ_loans.rename(index=str, columns={'loan_purchase_num': 'count_loans'})

df_occ_loans = df_occ_loans.merge(df_occ_cnts, on='occupation')
df_occ_loans['loans_per_occ'] = df_occ_loans['count_loans'] / df_occ_loans['count_reg']
gt_than = 3
df_display = df_occ_loans[df_occ_loans['count_reg'] > gt_than].sort_values('loans_per_occ', ascending=False).head(30)

fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True)

sns.barplot(x='loans_per_occ', y='occupation', data=df_display, color='chocolate')

ax.set_title('Top 30 Occupations - Most Loan Contributions Per Occupation User; >' + str(gt_than) + ' Users With Occupation', fontsize=15)
ax.set(ylabel='Occupation', xlabel='Number of Loan Contributions')
plt.show()
gt_than = 50
df_display = df_occ_loans[df_occ_loans['count_reg'] > gt_than].sort_values('loans_per_occ', ascending=False).head(30)

fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True)

sns.barplot(x='loans_per_occ', y='occupation', data=df_display, color='limegreen')

ax.set_title('Top 30 Occupations - Most Loan Contributions Per Occupation User; >' + str(gt_than) + ' Users With Occupation', fontsize=15)
ax.set(ylabel='Occupation', xlabel='Number of Loan Contributions')
plt.show()
gt_than = 500
df_display = df_occ_loans[df_occ_loans['count_reg'] > gt_than].sort_values('loans_per_occ', ascending=False).head(30)

fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True)

sns.barplot(x='loans_per_occ', y='occupation', data=df_display, color='cornflowerblue')

ax.set_title('Top 30 Occupations - Most Loan Contributions Per Occupation User; >' + str(gt_than) + ' Users With Occupation', fontsize=15)
ax.set(ylabel='Occupation', xlabel='Number of Loan Contributions')
plt.show()
gt_than = 1500
df_display = df_occ_loans[df_occ_loans['count_reg'] > gt_than].sort_values('loans_per_occ', ascending=False).head(30)

fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True)

sns.barplot(x='loans_per_occ', y='occupation', data=df_display, color='mediumaquamarine')

ax.set_title('Top 30 Occupations - Most Loan Contributions Per Occupation User; >' + str(gt_than) + ' Users With Occupation', fontsize=15)
ax.set(ylabel='Occupation', xlabel='Number of Loan Contributions')
plt.show()
df_display = df_exp.merge(df_loans[['loan_id', 'posted_month']], on='loan_id')[['permanent_name', 'posted_month']].groupby(['permanent_name', 'posted_month']).size().reset_index(name='counts').groupby(['posted_month']).size().reset_index(name='counts')

fig, ax = plt.subplots(1, 1, figsize=(20, 8), sharex=True)

plt.plot(df_display['posted_month'], df_display['counts'])
#plt.legend(['LBP', 'USD'], loc='upper left')
ax.set_title('Monthly Active Users (at least one loan contribution within month)', fontsize=15)
plt.show()
#need memory
del df_cntry_sum
del df_lender_sum
del df_lender_cntry
del df_cntry_cnts
del df_lender_state
del df_state_cnts
del df_state_sum
del df_lender_sum_state
gc.collect()
# merge exploaded users to loans to get posted times
df_display = df_exp.merge(df_loans[['loan_id', 'posted_time']], on='loan_id')[['permanent_name', 'posted_time']].drop_duplicates().sort_values(['permanent_name', 'posted_time'])

# get a distinct list of names and loan start dates
#df_last_visit = df_loans_users[(df_loans_users['permanent_name'] == 'sam4326') | (df_loans_users['permanent_name'] == 'rebecca3499')][['permanent_name', 'posted_time']].drop_duplicates().sort_values(['permanent_name', 'posted_time'])
#df_display = df_loans_users[['permanent_name', 'posted_time']].drop_duplicates() #.sort_values(['permanent_name', 'posted_time'])

# get the prior loan date for user
df_display['prev_loan_dt'] = df_display.groupby('permanent_name')['posted_time'].shift()

df_display.dropna(axis=0, how='any', inplace=True)

# calc days different
df_display['date_diff'] = (df_display['posted_time'] - df_display['prev_loan_dt']).dt.days
df_disp = df_display.groupby('posted_time')['date_diff'].mean().to_frame()
df_disp.reset_index(level=0, inplace=True)

df_disp = df_disp[df_disp['posted_time'] <= '2017-12-25']

fig, ax = plt.subplots(1, 1, figsize=(20, 8), sharex=True)

plt.plot(df_disp['posted_time'], df_disp['date_diff'])

ax.set_title('Average Days Since Last Loan Visit', fontsize=15)
plt.show()
df_disp = df_display.groupby('posted_time')['date_diff'].median().to_frame()
df_disp.reset_index(level=0, inplace=True)

df_disp = df_disp[df_disp['posted_time'] <= '2017-12-25']

fig, ax = plt.subplots(1, 1, figsize=(20, 8), sharex=True)

plt.plot(df_disp['posted_time'], df_disp['date_diff'])

ax.set_title('Median Days Since Last Loan Visit', fontsize=15)
plt.show()
df_display = df_loans[df_loans['status'] != 'fundRaising'].groupby('posted_time')[['funded_amount', 'loan_amount']].sum()
df_display.reset_index(level=0, inplace=True)
df_display['posted_year'] = df_display['posted_time'].dt.year
df_display['gap_amount'] = df_display['loan_amount'] - df_display['funded_amount']
df_display['day_of_year'] = df_display['posted_time'].dt.dayofyear
df_display['month_of_year'] = df_display['posted_time'].dt.month
fig, lst = plt.subplots(4, 1, figsize=(20, 12), sharex=False)
j = 2014

for i in lst:

    i.plot(df_display[df_display['posted_year'] == j]['posted_time'], df_display[df_display['posted_year'] == j]['loan_amount'], color='#67c5cb', label='loan_amount')
    i.plot(df_display[df_display['posted_year'] == j]['posted_time'], df_display[df_display['posted_year'] == j]['funded_amount'], color='#cb6d67', label='funded_amount')
    i.plot(df_display[df_display['posted_year'] == j]['posted_time'], df_display[df_display['posted_year'] == j]['gap_amount'], color='salmon', label='gap_amount')
    j = j+1

lst[0].set_title('Funding Gap By Day; 2014-2017', fontsize=15)
lst[0].legend(loc='upper left', frameon=True)
    
plt.show()
df_disp = df_display.groupby('day_of_year')['gap_amount'].agg(['sum', 'count']).reset_index()
df_disp2 = df_display.groupby('month_of_year')['gap_amount'].agg(['sum', 'count']).reset_index()
df_disp2['gap_per_loan'] = df_disp2['sum'] / df_disp2['count']
fig, (ax1, ax2, ax5) = plt.subplots(3, 1, figsize=(20, 14), sharex=False)

ax1.plot(df_disp['day_of_year'], df_disp['sum'], color='salmon')
ax1.set_title('Funding Gap 2014-2017 by Day of Year', fontsize=15)
#ax3 = ax1.twinx()
#ax3.plot(df_disp['day_of_year'], df_disp['count'], color='green')

ax2.plot(df_disp2['month_of_year'], df_disp2['sum'], color='salmon', label='funding gap')
ax2.set_title('Funding Gap 2014-2017 by Month of Year vs. New Loan Requests', fontsize=15)
ax4 = ax2.twinx()
ax4.plot(df_disp2['month_of_year'], df_disp2['count'], color='darkblue', label='\nnew loan requests')
ax2.legend(loc='upper left', frameon=True)
leg = ax4.legend(loc='upper left', frameon=True)
leg.get_frame().set_alpha(0)

ax5.plot(df_disp2['month_of_year'], df_disp2['gap_per_loan'], color='c')
ax5.set_title('Funding Gap Per Loan, Aggregate 2014-2017 by Month of Year', fontsize=15)

plt.show()
