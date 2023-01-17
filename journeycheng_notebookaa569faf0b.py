# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/clicks_train.csv')

df_test = pd.read_csv('../input/clicks_test.csv')
df_train.shape
df_train.head()
print('the number of unique display_id: ' + str(len(df_train['display_id'].unique())))

print('the number of unique ad_id: '+ str(len(df_train['ad_id'].unique())))
ad_in_display = df_train.groupby('display_id')['ad_id'].count().value_counts()

sns.barplot(ad_in_display.index, ad_in_display.values)
size_display.barplot()
ad_usage_train = df_train.groupby('ad_id')['ad_id'].count()
page_view_user_country = pd.read.csv('../input/page_views_sample.csv', usecpls=['uuid', 'geo_location'])