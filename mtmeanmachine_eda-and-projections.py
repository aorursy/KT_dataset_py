# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from warnings import filterwarnings

filterwarnings("ignore")

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):





# Any results you write to the current directory are saved as output.
#Use for filters later

initials = ['Confirmed','Deaths','Recovered']

pct = ['Confirmed_Change','Death_Change','Recover_Change']

log = ['Confirmed_Log','Death_Log','Recover_Log']

rates = ['MortalityRate','RecoveryRate']

original_columns = ['SNo', 'ObservationDate', 'Province/State', 'Country/Region','Last Update', 'Confirmed', 'Deaths', 'Recovered', 'Date']





df_all = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv',parse_dates=['ObservationDate'])

df_all['Date'] = df_all['ObservationDate'].apply(lambda x: pd.to_datetime(pd.Timestamp(x).date()))

df_health = pd.read_csv("/kaggle/input/health-nutrition-and-population-statistics/data.csv")





df_totals = df_all.pivot_table(index="Date",values=['Confirmed','Deaths','Recovered'],aggfunc="sum")

df_totals[pct] = df_totals[['Confirmed','Deaths','Recovered']].pct_change()

df_totals[log] = df_totals[initials].apply(lambda x: np.log(x))

df_totals['MortalityRate'] = df_totals['Deaths'] / df_totals['Confirmed']

df_totals['RecoveryRate'] = df_totals['Recovered'] / df_totals['Confirmed']
def build_subset(df,group,subset):

    df[group] = df[group].replace([np.nan,np.NaN,np.NAN],"")

    df_filtered = df[df[group].str.contains(subset)]

    df_totals = df_filtered.pivot_table(index="Date",values=['Confirmed','Deaths','Recovered'],aggfunc="sum")

    df_totals[pct] = df_totals[initials].pct_change()

    df_totals[log] = df_totals[initials].apply(lambda x: np.log(x))

    df_totals['MortalityRate'] = df_totals['Deaths'] / df_totals['Confirmed']

    df_totals['RecoveryRate'] = df_totals['Recovered'] / df_totals['Confirmed']

    return df_totals, df_filtered



def plot_changes(df,location):

    fig, ax = plt.subplots(2,2)

    df[pct].plot(figsize=(24,8),ax=ax[0,0],title="{} Daily Percent Changes".format(location),sharex=True,sharey=False)

    df[log].plot(figsize=(24,8),ax=ax[0,1],title="{} Daily Changes - Log Scale".format(location))

    df[initials].plot(figsize=(24,8),ax=ax[1,0],title="{} Daily Changes".format(location))

    df[rates].plot(figsize=(24,8),ax=ax[1,1],title="{} Daily Rate Changes".format(location))

    ax[0,0].axhline(0,c="black")

    ax[1,1].axhline(0,c="black")

    plt.show()



us_total, df_us = build_subset(df_all,"Country/Region","US")

plot_changes(us_total,"US")
sk, df_sk = build_subset(df_all,"Country/Region","South Korea")

plot_changes(sk,"South Korea")