# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt
import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
sns.set(font_scale=1.2)
df = pd.read_csv('/kaggle/input/uncover/UNCOVER/johns_hopkins_csse/2019-novel-coronavirus-covid-19-2019-ncov-data-repository-confirmed-cases.csv')
df.head()
df['date'] = pd.to_datetime(df['date'])
df.dtypes
plt.figure(figsize=(20, 8))
sns.lineplot(x="date", y="confirmed", data=df.groupby(['date'], as_index=False).agg({'confirmed': 'sum'}))
max_count_counties = ['China', 'Italy', 'US', 'Spain', 'Germany', 'Iran', 'France',
       'Korea, South', 'United Kingdom', 'Netherlands',
       'Pakistan', 'Russia', 'India']

plt.figure(figsize=(20, 8))
chart = sns.lineplot(x="date", y="confirmed", hue='country_region',
             data=df[df['country_region'].isin(max_count_counties)].groupby(['date', 'country_region'], as_index=False).agg({'confirmed': 'sum'}))
