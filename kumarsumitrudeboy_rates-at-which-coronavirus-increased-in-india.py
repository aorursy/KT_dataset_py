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
df = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")
df.head()
df_new = df.melt(id_vars=["Province/State","Country/Region","Lat","Long"],value_name="Count", var_name="Date").reset_index()
df_new.head()
df_new.drop("index", axis=1)
df_new.isnull().sum()
df_new.dtypes
#df_new["Date"] = df_new["Date"].to_datetime(format="%d/%m/%Y")
df_new["Date"] = pd.to_datetime(df_new.Date, infer_datetime_format=True)
df_new.head()
df_india = df_new[df_new["Country/Region"] == "India"]
df_india.head()
df_india["rate"] = (lambda x: df_india["Date"][x]-df_india["Date"][x-1] for x in df_india.index)
df_india.drop("rate", axis=1)
df_india = df_india.reset_index()
df_india = df_india.drop(["level_0","index"], axis=1)
df_india.head()
df_india.head()
df_india = df_india.reset_index()
df_india.head()
rates = []

for x in df_india["index"]:

    if x > 1: 

        rate = df_india.iloc[x, 6]-df_india.iloc[x-1,6]

        rates.append(rate)

    else:

        rate = df_india.iloc[x, 6]

        rates.append(rate)
print(rates)
len(rates)
df_india.count
df_india["rate"] = rates
df_india.head()
import matplotlib.pyplot as plt
#df_india.plot(df_india["Date"],df_india["rate"])

df_india.Date.freq = 'd' 

df_india.set_index("Date")
index_val = df_india.loc[df_india.rate == 121]

index_val
from matplotlib import rcParams



rcParams["figure.figsize"] = [20, 10]

fig = plt.figure()



ax = fig.add_axes([.1,.1,1,1])



ax.set_xlabel("Date")

ax.set_ylabel("Increase/Day")



ax.set_xticks(range(0,66))

ax.set_xticklabels(df_india.Date, rotation=60, fontsize='medium')

ax.grid()



plt.plot(df_india["rate"])



ax.annotate("Maximum increase in a day", xy=(63,121), xytext=(65,121), arrowprops=dict(facecolor='salmon', shrink=0.05))

plt.show()
#So we can infer on lockdown rates of cases of confirmed coronaviruses decreased.