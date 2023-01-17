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
import numpy as np

import pandas as pd

import seaborn as sns 

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

import fbprophet # Necessary for forecasting.
df = pd.read_csv('../input/covid19turkey/Covid19-Turkey.csv')
#Adding number of days column.

number_of_days = pd.DataFrame(np.arange(1,len(df.Date)+1,1))

a = {"Number Of Days": number_of_days.values}

df = df.assign(**a)

df.head()
#Adding daily deaths column.

yesterday_deaths = 0

Daily_deaths = []

for current_deaths in df['Total Deaths']:

    if current_deaths>yesterday_deaths:

        Daily_deaths.append(current_deaths-yesterday_deaths)

    else :

        Daily_deaths.append(0)

    yesterday_deaths = current_deaths

Daily_deaths=pd.DataFrame(Daily_deaths)

df['Daily Deaths'] = Daily_deaths
#Adding daily recovered column.

yesterday_recovered = 0

Daily_recovered = []

for current_recovered in df['Total Recovered']:

    if current_recovered>yesterday_recovered:

        Daily_recovered.append(current_recovered-yesterday_recovered)

    else :

        Daily_recovered.append(0)

    yesterday_recovered = current_recovered

Daily_recovered=pd.DataFrame(Daily_recovered)

df['Daily Recovered'] = Daily_recovered
sns.lineplot(x="Number Of Days", y="Total Cases", data = df)
sns.lineplot(x="Number Of Days", y="Daily Cases", data = df)
sns.lineplot(x="Number Of Days", y="Total Deaths", data = df)
sns.lineplot(x="Number Of Days", y="Daily Deaths", data = df)
plt.plot(df['Number Of Days'],df['Daily Test Cases'],color ='blue',label ='Daily Test Cases')

plt.plot(df['Number Of Days'],df['Daily Cases'],color ='red',label='Daily Cases')

plt.legend()

plt.xlabel('Number Of Days')

plt.ylabel('Value')
plt.plot(df['Number Of Days'],df['Daily Recovered'],color ='blue',label ='Daily Recovered')

plt.plot(df['Number Of Days'],df['Daily Cases'],color ='red',label='Daily Cases')

plt.plot(df['Number Of Days'],df['Total Cases'],color ='green',label ='Total Cases')

plt.legend()

plt.xlabel('Number Of Days')

plt.ylabel('Value')
tc=df['Total Cases']

nod=df['Number Of Days']

date = df["Date"]

date = date.str.replace("/","-")

tc_nod = pd.DataFrame({"Total Cases": tc,"Date": date})

td=df['Total Deaths']

td_nod = pd.DataFrame({"Total Deaths": td,"Date": date})

tr=df['Total Recovered']

tr_nod = pd.DataFrame({"Total Recovered": tr,"Date": date})

dc=df['Daily Cases']

dc=pd.DataFrame({"Daily Cases": dc,"Date": date})
tc_nod = tc_nod.rename(columns={'Date': 'ds', 'Total Cases': 'y'})

fbp1 = fbprophet.Prophet()

fbp1.fit(tc_nod)
future1 = fbp1.make_future_dataframe(periods=30,freq="M")

future1.tail()
forecast1 = fbp1.predict(future1)

forecast1[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = fbp1.plot(forecast1)

plt.xlabel('Days')

plt.ylabel('Total Cases')

plt.ticklabel_format(style='plain', axis='y')
td_nod = td_nod.rename(columns={'Date': 'ds', 'Total Deaths': 'y'})

fbp2 = fbprophet.Prophet()

fbp2.fit(td_nod)
future2 = fbp2.make_future_dataframe(periods=30,freq="M")

future2.tail()
forecast2 = fbp2.predict(future2)

forecast2[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig2 = fbp2.plot(forecast2)

plt.xlabel('Date')

plt.ylabel('Total Deaths')

plt.ticklabel_format(style='plain', axis='y')
tr_nod = tr_nod.rename(columns={'Date': 'ds', 'Total Recovered': 'y'})

fbp3 = fbprophet.Prophet()

fbp3.fit(tr_nod)
future3 = fbp3.make_future_dataframe(periods=30,freq="M")

future3.tail()
forecast3 = fbp3.predict(future3)

forecast3[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig3 = fbp3.plot(forecast3)

plt.xlabel('Days')

plt.ylabel('Total Recovered')

plt.ticklabel_format(style='plain', axis='y')
dc = dc.rename(columns={'Date': 'ds', 'Daily Cases': 'y'})

fbp4 = fbprophet.Prophet()

fbp4.fit(dc)
future4 = fbp4.make_future_dataframe(periods=100,freq="D")

future4.tail()
forecast4 = fbp4.predict(future4)

forecast4[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig4 = fbp4.plot(forecast4)

plt.xlabel('Days')

plt.ylabel('Daily Cases')