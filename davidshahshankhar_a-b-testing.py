# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

#Examine the first few rows of ad_clicks.
ad_clicks = pd.read_csv('/kaggle/input/ad-clicks/ad_clicks.csv')
print(ad_clicks.head())




most_views = ad_clicks.groupby('utm_source').count().user_id.reset_index()
print(most_views)
ad_clicks['is_click'] = ~ad_clicks['ad_click_timestamp'].isnull()
print(ad_clicks.head())

clicks_by_source = ad_clicks.groupby(['utm_source','is_click']).user_id.count().reset_index()
print(clicks_by_source)

clicks_pivot =clicks_by_source.pivot(
  columns ='is_click',
  index ='utm_source',
  values='user_id'
).reset_index()
print(clicks_pivot)
clicks_pivot['percent_clicked'] = clicks_pivot[True] / (clicks_pivot[True] +  clicks_pivot[False])*100
print(clicks_pivot)
print(ad_clicks.groupby('experimental_group').user_id.count().reset_index())

print(ad_clicks.groupby(['experimental_group','is_click']).user_id.count().reset_index().pivot(
  columns = 'is_click',
  index ='experimental_group',
  values ='user_id'
).reset_index())
a_clicks = ad_clicks[ad_clicks.experimental_group == 'A']
b_clicks = ad_clicks[ad_clicks.experimental_group == 'B']


a_clicks_pivot= a_clicks.groupby(['is_click','day']).user_id.count().reset_index().pivot(
  columns = 'is_click',
  index ='day',
  values = 'user_id'
).reset_index()

a_clicks_pivot['percentage']=a_clicks_pivot[True]/(a_clicks_pivot[True]+a_clicks_pivot[False])
print(a_clicks_pivot)

b_clicks_pivot= b_clicks.groupby(['is_click','day']).user_id.count().reset_index().pivot(
  columns = 'is_click',
  index ='day',
  values = 'user_id'
).reset_index()

b_clicks_pivot['percentage']=b_clicks_pivot[True]/(b_clicks_pivot[True]+b_clicks_pivot[False])
print(b_clicks_pivot)