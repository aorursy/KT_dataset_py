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

        pass

        # print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv', parse_dates=['ObservationDate'])
print('Last observation date: %s' % df.ObservationDate.value_counts().sort_index(ascending=False).index[0])
by_country = df.groupby(['Country/Region', 'ObservationDate']).sum()

cases = by_country.loc[['Mainland China']][['Confirmed', 'Deaths']]

cases = pd.concat([cases, cases.pct_change(periods=7) + 1], axis=1)

cases.columns = ['Confirmed', 'Deaths', 'Confirmed_7day_change_rate', 'Deaths_7day_change_rate']

print(cases[['Confirmed', 'Confirmed_7day_change_rate']])

cases['Confirmed'].plot(logy=True)