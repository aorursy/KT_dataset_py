# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")

test = pd.read_csv("../input/covid19-global-forecasting-week-2/test.csv")

submit = pd.read_csv("../input/covid19-global-forecasting-week-2/submission.csv")
data.head()
data.shape
data.Country_Region.nunique()
data.Country_Region.value_counts()
data.Country_Region[data.Country_Region=="India"].value_counts()
print("Start date:",min(data.Date))

print("End date:",max(data.Date))
min(data.Date).split("-")
data.head()
# data.groupby(data.Country_Region)
# a = data.groupby(['Country_Region', 'ConfirmedCases'])
data.groupby(['Country_Region']).agg({'ConfirmedCases':['sum']})
confirmed_total = data.groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total = data.groupby(['Date']).agg({'Fatalities':['sum']})
total = confirmed_total.join(fatalities_total)
fig, (axis1, axis2) = plt.subplots(1,2, figsize=(17,8))

axis1.set_xlabel("Date",size=20)

axis1.set_ylabel("Number of cases",size=20)

axis1.set_title("Global CONFIRMED cases",size=20)



axis2.set_xlabel("Date",size=20)

axis2.set_ylabel("Number of cases",size=20)

axis2.set_title("Global FATALITIES cases",size=20)



confirmed_total.plot(ax=axis1, color='orange')

fatalities_total.plot(ax=axis2, color='blue')
confirmed_total_india = data[data.Country_Region == 'India'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_india = data[data.Country_Region == 'India'].groupby(['Date']).agg({'Fatalities':['sum']})



total_india = confirmed_total_india.join(fatalities_total_india)
total_india
fig, (axis1, axis2) = plt.subplots(1,2, figsize=(17,8))

axis1.set_xlabel("Date",size=20)

axis1.set_ylabel("Number of cases",size=20)

axis1.set_title("Global CONFIRMED cases(INDIA)",size=20)



axis2.set_xlabel("Date",size=20)

axis2.set_ylabel("Number of cases",size=20)

axis2.set_title("Global FATALITIES cases(INDIA)",size=20)



confirmed_total_india.plot(ax=axis1, color='orange')

fatalities_total_india.plot(ax=axis2, color='blue')
confirmed_total_china = data[data.Country_Region == 'China'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_china = data[data.Country_Region == 'China'].groupby(['Date']).agg({'Fatalities':['sum']})



total_china = confirmed_total_china.join(fatalities_total_china)
fig, (axis1, axis2) = plt.subplots(1,2, figsize=(17,8))

axis1.set_xlabel("Date",size=20)

axis1.set_ylabel("Number of cases",size=20)

axis1.set_title("Global CONFIRMED cases(CHINA)",size=20)



axis2.set_xlabel("Date",size=20)

axis2.set_ylabel("Number of cases",size=20)

axis2.set_title("Global FATALITIES cases(CHINA)",size=20)



confirmed_total_china.plot(ax=axis1, color='orange')

fatalities_total_china.plot(ax=axis2, color='blue')
confirmed_total_italy = data[data.Country_Region == 'Italy'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_italy = data[data.Country_Region == 'Italy'].groupby(['Date']).agg({'Fatalities':['sum']})



total_italy = confirmed_total_italy.join(fatalities_total_italy)
fig, (axis1, axis2) = plt.subplots(1,2, figsize=(17,8))

axis1.set_xlabel("Date",size=20)

axis1.set_ylabel("Number of cases",size=20)

axis1.set_title("Global CONFIRMED cases(ITALY)",size=20)



axis2.set_xlabel("Date",size=20)

axis2.set_ylabel("Number of cases",size=20)

axis2.set_title("Global FATALITIES cases(ITALY)",size=20)



confirmed_total_italy.plot(ax=axis1, color='blue',linewidth=6)

fatalities_total_italy.plot(ax=axis2, color='red',linewidth=6)