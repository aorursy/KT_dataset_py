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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('seaborn-dark')
df = pd.read_csv('/kaggle/input/flight-delay-prediction/Jan_2019_ontime.csv')

df = df.drop('Unnamed: 21', axis=1)

df.head()
df.drop('DEP_TIME_BLK', axis=1, inplace=True)

df.drop('TAIL_NUM', axis=1, inplace=True)

df.drop('OP_CARRIER', axis=1, inplace=True)

df['DEP_TIME'].fillna(-1, inplace=True)

df['DEP_DEL15'].fillna(-1, inplace=True)

df['ARR_DEL15'].fillna(-1, inplace=True)

df['ARR_TIME'].fillna(-1, inplace=True)

df['DAY_OF_MONTH'] = pd.Categorical(df['DAY_OF_MONTH'])

df['DAY_OF_WEEK'] = pd.Categorical(df['DAY_OF_WEEK'])

df['OP_CARRIER_AIRLINE_ID'] = pd.Categorical(df['OP_CARRIER_AIRLINE_ID'])

df['OP_CARRIER_FL_NUM'] = pd.Categorical(df['OP_CARRIER_FL_NUM'])

df['ORIGIN_AIRPORT_ID'] = pd.Categorical(df['ORIGIN_AIRPORT_ID'])

df['ORIGIN_AIRPORT_SEQ_ID'] = pd.Categorical(df['ORIGIN_AIRPORT_SEQ_ID'])

df['DEST_AIRPORT_ID'] = pd.Categorical(df['DAY_OF_WEEK'])

df['DEST_AIRPORT_SEQ_ID'] = pd.Categorical(df['DEST_AIRPORT_SEQ_ID'])

df['DEP_DEL15'] = pd.Categorical(df['DEP_DEL15'])

df['ARR_DEL15'] = pd.Categorical(df['ARR_DEL15'])

# df['CANCELLED'] = pd.Categorical(df['CANCELLED'])

df['DIVERTED'] = pd.Categorical(df['DIVERTED'])

df['ARR_TIME'] = df['ARR_TIME'].astype('str').str[:-2].apply(lambda x: '0' + x if len(x) < 4 else x)

df['DEP_TIME'] = df['DEP_TIME'].astype('str').str[:-2].apply(lambda x: '0' + x if len(x) < 4 else x)
fig, ax = plt.subplots(1, 2, figsize=(20, 5))

ax[0].bar(np.arange(0,31), df['CANCELLED'].groupby(df['DAY_OF_MONTH']).count())

ax[0].set_xticks(np.arange(0, 31));

ax[0].set_xticklabels(np.arange(1, 32));

ax[0].set_title('Number of cancelled flights by day of month', fontsize=14)



ax[1].bar(np.arange(0, 7, ), df['CANCELLED'].groupby(df['DAY_OF_WEEK']).count())

ax[1].set_xticks(np.arange(0, 7));

ax[1].set_xticklabels(['Mon', 'Tue', 'Wen', 'Thu', 'Fri', 'Sat', 'Sun']);

ax[1].set_title('Number of cancelled flights by day of week', fontsize=14)
canc_flights = pd.DataFrame(df['OP_UNIQUE_CARRIER'].value_counts()).join(df['CANCELLED'].groupby(df['OP_UNIQUE_CARRIER']).sum())

frac_canc = canc_flights['CANCELLED']/canc_flights['OP_UNIQUE_CARRIER']

plt.figure(figsize=(15,4))

plt.bar(np.arange(len(frac_canc)), frac_canc)

plt.xticks(np.arange(len(frac_canc)), frac_canc.index);

plt.title('Fraction of cancelled flights by OP_UNIQUE_CARRIER')
plt.figure(figsize=(15, 4))

plt.bar(np.arange(1), df['CANCELLED'].groupby(df['DEP_TIME'].astype('str').str[:2]).count()[0], hatch='/', alpha=.4)

plt.bar(np.arange(1, 25), df['CANCELLED'].groupby(df['DEP_TIME'].astype('str').str[:2]).count()[1:])

plt.xticks(np.arange(25), np.arange(0, 25));

plt.text(-0.8, 18000, 'NA values');

plt.title('Cancels distribution over hours')