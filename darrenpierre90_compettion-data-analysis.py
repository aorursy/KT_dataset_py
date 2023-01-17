#https://towardsdatascience.com/time-based-cross-validation-d259b13d42b8

#https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

from pandas_profiling import ProfileReport

df=pd.read_csv('/kaggle/input/covid-global-forcast.csv',parse_dates=["Date"])
df.info()

df["Province/State"]=df["Province/State"].fillna("")
df.head(10)
df["Province/State"]=df["Province/State"].fillna("")

df=df.sort_values(by=["Date","Country/Region","Province/State"])

max_date=max(df["Date"])

min_date=min(df["Date"])

print(f"Date ranges from {min_date} to {max_date}")
profile=ProfileReport(df)

profile.to_widgets()
df_line=pd.DataFrame()

df_date=df.groupby(["Date"]).sum().reset_index()

df_date.head(9)

cols=["# ConfirmedCases","# Fatalities","# Recovered_cases"]

for col in cols:

    df_date[col]=np.log(df_date[col])



ax = sns.lineplot(x="Date", y=cols[0], data=df_date)



a = sns.lineplot(x="Date", y=cols[1], data=df_date)

ax = sns.lineplot(x="Date", y=cols[2], data=df_date)