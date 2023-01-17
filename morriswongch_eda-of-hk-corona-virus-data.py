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
import json

import requests
df = pd.read_csv('/kaggle/input/hk-corona-virus-case/covid-19-case-hk.csv', index_col=0)
source = requests.get('https://wars.vote4.hk/page-data/en/cases/page-data.json').content

total_no_cases = len(json.loads(source.decode("utf-8"))['result']['data']['allWarsCase']['edges'])



all_case_list = [json.loads(source.decode("utf-8"))['result']['data']['allWarsCase']['edges'][i]['node'] for i in range(0, total_no_cases)]



all_case_df = pd.DataFrame(all_case_list)

all_case_df['case_no'] = all_case_df['case_no'].astype('int')

all_case_df['age'] = all_case_df['age'].astype('int')



df = all_case_df.sort_values(by=['confirmation_date'])
df['classification'].value_counts().plot.pie(autopct='%1.1f%%', fontsize=15, figsize=(8,8), pctdistance=0.85)
df['confirmation_date'].value_counts().sort_index().cumsum().plot(figsize=(10,5)).grid(zorder=0)
cumsum_df = pd.DataFrame(columns=['confirmation_date'] + list(df['classification'].unique()))

cumsum_df['confirmation_date'] = sorted(df['confirmation_date'].unique())
for date in sorted(df['confirmation_date'].unique()):

    date_filter = df['confirmation_date'] == date

    cumsum_date_filter = cumsum_df['confirmation_date'] == date



    tmp_df = pd.DataFrame(df[date_filter][['confirmation_date', 'classification']]['classification'].value_counts()).T.reset_index(drop=True)

    tmp_df['confirmation_date'] = date



    for class_type in list(tmp_df.columns[:-1]):

        cumsum_df.loc[cumsum_date_filter, class_type] = int(tmp_df[class_type].values[0])
cumsum_df.fillna(method='ffill').fillna(0).cumsum().plot(figsize=(10,5))
cumsum_df.fillna(method='ffill').fillna(0).plot(figsize=(10,5))
df['status'].value_counts().plot.pie(autopct='%1.1f%%', fontsize=15, figsize=(8,8), legend=True, pctdistance=0.85)
df['age'].plot.hist(figsize=(10,5)).grid(zorder=0)
df['age'].plot(kind='kde', figsize=(10,5)).grid(zorder=0)
df['hospital_en'].value_counts().sort_values().plot.barh(figsize=(10,5),fontsize=15)
discharge_filter = df['status'] == 'discharged'
df[discharge_filter]['age'].plot(kind='kde', figsize=(10,5), legend=True).legend(["discharge", "hospitalized"]);

df[~discharge_filter]['age'].plot(kind='kde', figsize=(10,5), legend=True).legend(["discharge", "hospitalized"]);