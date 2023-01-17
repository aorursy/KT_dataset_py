# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

masterdata = pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")

confirmed = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_2019_ncov_confirmed.csv")

ncov_deaths = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_2019_ncov_deaths.csv")

ncov_recovered = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_2019_ncov_recovered.csv")
masterdata.columns
masterdata=masterdata.rename(columns={'Province/State':'Province'})

masterdata.Date=masterdata.Date.apply(lambda df:df.split()[0])
masterdata.Province.unique()
data_cn=masterdata.loc[(data.Country=='China') | (data.Country=='Mainland China')].sort_values(by='Date',ascending=True)

Wuhan_confirmed=data_cn.loc[(data_cn.Province=='Hubei') ,['Date','Confirmed']]
Wuhan_confirmed=Wuhan_confirmed.set_index('Date')
Wuhan_confirmed.head()
plt.figure(figsize=(20,10))

sns.lineplot(data=Wuhan_confirmed)
Num_Confirmed=list(Wuhan_confirmed.Confirmed)

Growth_rate=[((Num_Confirmed[i]-Num_Confirmed[i-1])/Num_Confirmed[i-1]) for i in range(1,21)]
print(Growth_rate)
print(Num_Confirmed)
b=[i for i in range(20)]
print(b)
sns.regplot(x=b,y=Growth_rate)
for i in range(20,70):

    a=(0.45-0.02*(i-1))

    Growth_rate.append(a)

    Num_Confirmed.append(Num_Confirmed[i-1]*(1+a))
print(Growth_rate)
print(Num_Confirmed)
c=[i for i in range(71)]

Estimation=pd.DataFrame({'Confirmed':Num_Confirmed},index=c)
sns.lineplot(data=Estimation)