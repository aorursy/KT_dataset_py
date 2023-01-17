# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



import datetime



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
patient = pd.read_csv('../input/coronavirusdataset/patient.csv')

trend = pd.read_csv('../input/coronavirusdataset/trend.csv')

time = pd.read_csv('../input/coronavirusdataset/time.csv')

route = pd.read_csv('../input/coronavirusdataset/route.csv')

patient.head(12)
route[(route['date'] < '2020-02-01') & ( route['province'] == 'Seoul')].groupby('city')['patient_id'].nunique()
fp_01 = pd.read_csv("../input/seoul-floating-population-2020/fp_2020_01_english.csv")
fp_01['date'] = fp_01['date'].apply(lambda x: datetime.datetime.strptime(x, "%Y.%m.%d").date()).astype('str')

fp_01['date'] = fp_01['date'].apply(lambda x: x[8:]) ## only use day



fp_01 = fp_01.sort_values(['date', 'hour', 'birth_year', 'sex'])  ## this data is not sorted.

fp_01.reset_index(drop= True, inplace = True)

fp_01.head()
patient['confirmed_date'][0]
print("first Infected date in korea: ", patient['confirmed_date'][0])
## Date when the visitor went to Gangnam-gu

sorted(list(route[route['city'] == 'Gangnam-gu']['date'].unique()))
tmp = fp_01[(fp_01['city'] == 'Gangnam-gu')]

fig, ax = plt.subplots(figsize=(14, 10))

sns.lineplot(data=tmp, x='date', y='fp_num',hue = 'birth_year', ax=ax)

plt.axvline(x = '20', color = 'b', ls = '--', alpha = 0.6) 

plt.axvline(x = '22', color = 'r', ls = '--', alpha = 0.6) 

plt.title('The patient"s first visit to Gangnam-gu is 22 days')

plt.show()
tmp = fp_01[(fp_01['city'] == 'Gangnam-gu')]

fig, ax = plt.subplots(figsize=(14, 10))

sns.lineplot(data=tmp, x='date', y='fp_num',hue = 'sex', ax=ax)

plt.axvline(x = '20', color = 'b', ls = '--', alpha = 0.6) 

plt.axvline(x = '22', color = 'r', ls = '--', alpha = 0.6) 

plt.title('The patient"s first visit to Gangnam-gu is 22 days')

plt.show()
## Date when the visitor went to Jongno-gu

sorted(list(route[route['city'] == 'Jongno-gu']['date'].unique()))
tmp = fp_01[(fp_01['city'] == 'Jongno-gu')]

fig, ax = plt.subplots(figsize=(14, 10))

sns.lineplot(data=tmp, x='date', y='fp_num',hue = 'birth_year', ax=ax)

plt.axvline(x = '20', color = 'b', ls = '--', alpha = 0.6) 

plt.axvline(x = '26', color = 'r', ls = '--', alpha = 0.6) 

plt.title('The patient"s first visit to Jongno-gu is 26 days')

plt.show()
tmp = fp_01[(fp_01['city'] == 'Jongno-gu')]

fig, ax = plt.subplots(figsize=(14, 10))

sns.lineplot(data=tmp, x='date', y='fp_num',hue = 'sex', ax=ax)

plt.axvline(x = '20', color = 'b', ls = '--', alpha = 0.6) 

plt.axvline(x = '26', color = 'r', ls = '--', alpha = 0.6) 

plt.title('The patient"s first visit to Jongno-gu is 26 days')

plt.show()
## Date when the visitor went to Jongno-gu

sorted(list(route[route['city'] == 'Jung-gu']['date'].unique()))
tmp = fp_01[(fp_01['city'] == 'Jung-gu') & (fp_01['date'] > '01')]

fig, ax = plt.subplots(figsize=(14, 10))

sns.lineplot(data=tmp, x='date', y='fp_num',hue = 'birth_year', ax=ax)

plt.axvline(x = '20', color = 'b', ls = '--', alpha = 0.6) 

plt.axvline(x = '20', color = 'r', ls = '--', alpha = 0.6) 

plt.title('The patient"s first visit to Jung-gu is 19 days')

plt.show()
tmp = fp_01[(fp_01['city'] == 'Jung-gu') & (fp_01['date'] > '01')]

fig, ax = plt.subplots(figsize=(14, 10))

sns.lineplot(data=tmp, x='date', y='fp_num',hue = 'sex', ax=ax)

plt.axvline(x = '20', color = 'b', ls = '--', alpha = 0.6) 

plt.axvline(x = '20', color = 'r', ls = '--', alpha = 0.6) 

plt.title('The patient"s first visit to Jung-gu is 19 days')

plt.show()
## Date when the visitor went to Jungnang-gu

sorted(list(route[route['city'] == 'Jungnang-gu']['date'].unique()))
tmp = fp_01[(fp_01['city'] == 'Jungnang-gu') & (fp_01['date'] > '01')]

fig, ax = plt.subplots(figsize=(14, 10))

sns.lineplot(data=tmp, x='date', y='fp_num',hue = 'birth_year', ax=ax)

plt.axvline(x = '20', color = 'b', ls = '--', alpha = 0.6) 

plt.axvline(x = '28', color = 'r', ls = '--', alpha = 0.6) 

plt.title('The patient"s first visit to Jungnang-gu is 28 days')

plt.show()
tmp = fp_01[(fp_01['city'] == 'Jungnang-gu') & (fp_01['date'] > '01')]

fig, ax = plt.subplots(figsize=(14, 10))

sns.lineplot(data=tmp, x='date', y='fp_num',hue = 'sex', ax=ax)

plt.axvline(x = '20', color = 'b', ls = '--', alpha = 0.6) 

plt.axvline(x = '28', color = 'r', ls = '--', alpha = 0.6) 

plt.title('The patient"s first visit to Jungnang-gu is 28 days')

plt.show()
def plot_dist_col(train_df, test_df, title ):

    '''plot dist curves for train and test weather data for the given column name'''

    train_df = pd.DataFrame(train_df.groupby('hour')['fp_num'].sum())

    train_df.reset_index(inplace = True)

    

    test_df = pd.DataFrame(test_df.groupby('hour')['fp_num'].sum())

    test_df.reset_index(inplace = True)

    

    fig, ax = plt.subplots(figsize=(10, 10))

    sns.lineplot(data=train_df, x='hour', y='fp_num', color='green', ax=ax).set_title('fp_num', fontsize=16)

    sns.lineplot(data=test_df, x='hour', y='fp_num', color='purple', ax=ax).set_title('fp_num', fontsize=16)

    plt.xlabel('hour', fontsize=16)

    plt.title(title, fontsize=20)

    plt.legend(['17day', '31day'])

    plt.show()
gan17 = fp_01[(fp_01['date'] == '17') & (fp_01['city'] == 'Gangnam-gu')]

gan17 = pd.DataFrame(gan17.groupby(['hour', 'birth_year'])['fp_num'].sum())

gan17.reset_index(inplace = True)
gan17
gan17 = fp_01[(fp_01['date'] == '17') & (fp_01['city'] == 'Gangnam-gu')]

gan31 = fp_01[(fp_01['date'] == '31') & (fp_01['city'] == 'Gangnam-gu')]

plot_dist_col(gan17, gan31, 'Gangnam-gu pattern')
gan17 = fp_01[(fp_01['date'] == '17') & (fp_01['city'] == 'Gangnam-gu') & (fp_01['birth_year'] == 20)]

gan31 = fp_01[(fp_01['date'] == '31') & (fp_01['city'] == 'Gangnam-gu') & (fp_01['birth_year'] == 20)]

plot_dist_col(gan17, gan31, 'Gangnam-gu 20 years old pattern')
gan17 = fp_01[(fp_01['date'] == '17') & (fp_01['city'] == 'Gangnam-gu')]

gan31 = fp_01[(fp_01['date'] == '31') & (fp_01['city'] == 'Gangnam-gu')]

plot_dist_col(gan17, gan31, 'Gangnam-gu pattern')
gan17 = fp_01[(fp_01['date'] == '17') & (fp_01['city'] == 'Jungnang-gu')]

gan31 = fp_01[(fp_01['date'] == '31') & (fp_01['city'] == 'Jungnang-gu')]

plot_dist_col(gan17, gan31, 'Jungnang-gu pattern')
gan17 = fp_01[(fp_01['date'] == '17') & (fp_01['city'] == 'Jungnang-gu') & (fp_01['birth_year'] == 20)]

gan31 = fp_01[(fp_01['date'] == '31') & (fp_01['city'] == 'Jungnang-gu') & (fp_01['birth_year'] == 20)]

plot_dist_col(gan17, gan31, 'Jungnang-gu 20 years old pattern')
gan17 = fp_01[(fp_01['date'] == '17') & (fp_01['city'] == 'Jongno-gu')]

gan31 = fp_01[(fp_01['date'] == '31') & (fp_01['city'] == 'Jongno-gu')]

plot_dist_col(gan17, gan31, 'Jongno-gu pattern')
gan17 = fp_01[(fp_01['date'] == '17') & (fp_01['city'] == 'Jongno-gu') & (fp_01['birth_year'] == 20)]

gan31 = fp_01[(fp_01['date'] == '31') & (fp_01['city'] == 'Jongno-gu') & (fp_01['birth_year'] == 20)]

plot_dist_col(gan17, gan31, 'Jongno-gu 20 years old pattern')
gan17 = fp_01[(fp_01['date'] == '17') & (fp_01['city'] == 'Jung-gu')]

gan31 = fp_01[(fp_01['date'] == '31') & (fp_01['city'] == 'Jung-gu')]

plot_dist_col(gan17, gan31, 'Jung-gu  pattern')
gan17 = fp_01[(fp_01['date'] == '17') & (fp_01['city'] == 'Jung-gu') & (fp_01['birth_year'] == 20)]

gan31 = fp_01[(fp_01['date'] == '31') & (fp_01['city'] == 'Jung-gu') & (fp_01['birth_year'] == 20)]

plot_dist_col(gan17, gan31, 'Jung-gu 20 years old pattern')