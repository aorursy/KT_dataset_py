# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
pd.options.display.max_colwidth = 400

from IPython.display import FileLink, FileLinks
train = pd.read_csv('/kaggle/input/web-traffic-time-series-forecasting/train_2.csv.zip')

keys = pd.read_csv('/kaggle/input/web-traffic-time-series-forecasting/key_2.csv.zip')

sub = pd.read_csv('/kaggle/input/web-traffic-time-series-forecasting/sample_submission_2.csv.zip')
train.head()
keys.head()
sub.head()
def split_page_col(page):

    tokens = page.split('_')

    article_name = ''.join(tokens[:-3])

    org = tokens[-3]

    access = tokens[-2]

    crawler = tokens[-1]

    return (article_name, org, access, crawler)



def split_page_col_wdate(page):

    tokens = page.split('_')

    article_name = ''.join(tokens[:-4])

    org = tokens[-4]

    access = tokens[-3]

    crawler = tokens[-2]

    date = tokens[-1]

    return (article_name, org, access, crawler,date)
keys['date'] = keys['Page'].apply(lambda x: x.split('_')[-1])

keys['Page'] = keys['Page'].apply(lambda x: '_'.join(x.split('_')[:-1]))

keys['date'] = pd.to_datetime(keys['date'], format='%Y-%m-%d')
keys.tail()
sub = sub.merge(keys, on='Id', how='left')
sub['date'] = pd.to_datetime(sub['date'], format='%Y-%m-%d')
print(sub['date'].min(), sub['date'].max())

print(sub['date'].max() - sub['date'].min())
print(sub['Page'].nunique())

print(train['Page'].nunique())
train.shape
train.iloc[:, 755:803]
print(sub.shape)

print(train.shape)
prev_year_data_cols = pd.date_range('2016-09-13', '2016-11-13')

train_flat = pd.melt(train.loc[:, ['Page'] + list(prev_year_data_cols.date.astype(str))], id_vars='Page', var_name='date')

train_flat['date'] = pd.to_datetime(train_flat['date'], format='%Y-%m-%d')
train_flat.head()
train_flat['prediction_date'] = train_flat['date'] + pd.DateOffset(years=1)

sub = sub[['Page', 'date', 'Id']].merge(train_flat[['Page', 'prediction_date', 'value']], left_on=('Page', 'date'), right_on=('Page', 'prediction_date'))

sub['value'] = sub['value'].fillna(0)
sub[['Id', 'value']].rename(columns={'value': 'visits'}).to_csv('all_submission.csv', index=False)

FileLink('all_submission.csv')
page_median = train.iloc[:, 1:].median(axis=1, skipna=True)
page_median = pd.DataFrame({'Page': train['Page'], 'median': page_median})
page_median.head()
sub_median = sub.merge(page_median, on='Page')[['Id', 'median']]
sub_median.isnull().mean()
sub_median
sub_median.rename(columns={'median': 'visits'}).to_csv('submission.csv', index=False)

FileLink('submission.csv')
prev_year_data_cols = pd.date_range('2016-09-13', '2016-11-13')

prev_year_median = train.loc[:, list(prev_year_data_cols.date.astype(str))].median(axis=1, skipna=True)

prev_year_median = pd.DataFrame({'Page': train['Page'], 'visits': prev_year_median})
sub_prev_year_median = sub.merge(prev_year_median, on='Page')[['Id', 'visits']]
sub_prev_year_median.isnull().mean()
sub_prev_year_median['visits'] = sub_prev_year_median['visits'].fillna(sub_prev_year_median['visits'].median())
# sub_prev_year_median.to_csv('submission_prev_year.csv',index=False)

# FileLink('submission_prev_year.csv')
median_60 = train.iloc[:, -60:].median(axis=1, skipna=True)

median_60 = pd.DataFrame({'Page': train['Page'], 'visits': median_60})

sub_median_60 = sub.merge(median_60, on='Page')[['Id', 'visits']]
sub_median_60.isnull().mean()
sub_median_60['visits'] = sub_median_60['visits'].fillna(0)
# sub_prev_year_median.to_csv('sub_median_60.csv',index=False)

# FileLink('sub_median_60.csv')