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
from pandas import DataFrame

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd





df=pd.read_csv(r"/kaggle/input/flight-delay-prediction/Jan_2020_ontime.csv")
df = df.drop('Unnamed: 21', axis=1)

df.head()
# The data consist of 17 different airlines.

df.OP_CARRIER.unique()

# We can see top 10 airlines and the percentage of delayed flight vs total flights.

df_crr=pd.DataFrame(df.groupby(["OP_UNIQUE_CARRIER","DEP_DEL15"])["ORIGIN_AIRPORT_ID"].count().unstack("DEP_DEL15").reset_index().values,columns=["Crr","Not_Delayed","Delayed"])

df_crr["Percent Of Delayed"]=df_crr.Delayed /( df_crr.Delayed+ df_crr.Not_Delayed)

df_crr.sort_values(by="Not_Delayed",ascending=False,inplace=True)

df_crr.head(10)
df_crr.set_index("Crr").plot(kind='bar',stacked=True)

plt.show()
# We can see top 10 origin airports and the percentage of delayed flight vs total flights.

df_org=pd.DataFrame(df.groupby(["ORIGIN","DEP_DEL15"])["ORIGIN_AIRPORT_ID"].count().unstack("DEP_DEL15").reset_index().values,columns=["Origin","Not_Delayed","Delayed"])

df_org["Percent Of Delayed"]=df_org.Delayed /( df_org.Delayed+ df_org.Not_Delayed)

df_org.sort_values(by="Not_Delayed",ascending=False,inplace=True)

df_org.head(10)
df_org.head(10).set_index("Origin").plot(kind='bar',stacked=True)

plt.show()
# We can see top 10 departure time blocks and the percentage of delayed flight vs total flights.

#It seems most of the flights are in the morning. But higher percentage of delay occurs after 1700.

df_depblk=pd.DataFrame(df.groupby(["DEP_TIME_BLK","DEP_DEL15"])["ORIGIN_AIRPORT_ID"].count().unstack("DEP_DEL15").reset_index().values,columns=["DEP_TIME_BLK","Not_Delayed","Delayed"])

df_depblk["PercentOfDelayed"]=df_depblk.Delayed /( df_depblk.Delayed+ df_depblk.Not_Delayed)

df_depblk.sort_values(by="PercentOfDelayed",ascending=False,inplace=True)

df_depblk.head(10)

df_depblk.sort_values(by="Not_Delayed",ascending=False,inplace=True)

df_depblk.head(10)

df_depblk.set_index("DEP_TIME_BLK").plot(kind='bar')



plt.show()
#In order to check if there is a relationship between distance and delay we can create a scatter plot. 

#From this plot we can see that there is not a clear relationship between distance and getting delays. 

#But we need to do logistic regression and check pseudo R^2 in order to have better understandig.

plt.scatter(df["DISTANCE"],df["DEP_DEL15"])

plt.show()
# Its clearly seen that most of the flights are on the 4th day. Nearly %20 of flights are delayed of 6th day.

df_dow=pd.DataFrame(df.groupby(["DAY_OF_WEEK","DEP_DEL15"])["ORIGIN_AIRPORT_ID"].count().unstack("DEP_DEL15").reset_index().values,columns=["DOW","Not_Delayed","Delayed"])

df_dow["PercentOfDelayed"]=df_dow.Delayed /( df_dow.Delayed+ df_dow.Not_Delayed)

df_dow.sort_values(by="Not_Delayed",ascending=False,inplace=True)

df_dow.head(10)
df_dow.set_index("DOW").plot(kind='bar')



plt.show()