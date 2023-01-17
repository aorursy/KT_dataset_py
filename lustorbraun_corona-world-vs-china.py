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
confirmed=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
confirmed.head(2)
confirmed.drop(['Province/State','Lat','Long'],inplace=True,axis=1)
confirmed.set_index('Country/Region',inplace=True)
confirmed=confirmed.transpose()

confirmed.reset_index(inplace=True)
confirmed.rename({'index':'Date'},axis='columns',inplace=True)
confirmed['Date']=pd.to_datetime(confirmed['Date'])
confirmed.sort_values(by=['Date'],inplace=True)
confirmed.head(2)
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
plt.figure(figsize=(18,6))

plt.plot(confirmed['China'],color='r')

plt.plot()
confirmed_notUSA=confirmed.drop('US',axis=1)
plt.figure(figsize=(18,12))

for n in range(1,266):

        plt.plot(confirmed_notUSA.iloc[:,n],color='b')

plt.plot(confirmed_notUSA['China'],color='r')

plt.plot()
recovered=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')
recovered.head(2)
recovered.drop(['Province/State','Lat','Long'],inplace=True,axis=1)

recovered.set_index('Country/Region',inplace=True)

recovered=recovered.transpose()

recovered.reset_index(inplace=True)

recovered.rename({'index':'Date'},axis='columns',inplace=True)

recovered['Date']=pd.to_datetime(recovered['Date'])

recovered.sort_values(by=['Date'],inplace=True)
recovered.columns
plt.figure(figsize=(18,6))

plt.plot(recovered['China'],color='r')

plt.plot()
plt.figure(figsize=(18,12))

for n in range(1,253):

        plt.plot(recovered.iloc[:,n],color='b')

plt.plot(recovered['China'],color='r')

plt.plot()
death=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')
death.head(2)
death.drop(['Province/State','Lat','Long'],inplace=True,axis=1)

death.set_index('Country/Region',inplace=True)

death=death.transpose()

death.reset_index(inplace=True)

death.rename({'index':'Date'},axis='columns',inplace=True)

death['Date']=pd.to_datetime(death['Date'])

death.sort_values(by=['Date'],inplace=True)
death.columns
plt.figure(figsize=(18,6))

plt.plot(death['China'],color='r')

plt.plot()
death.drop('US',axis=1,inplace=True)
plt.figure(figsize=(18,12))

for n in range(1,253):

        plt.plot(death.iloc[:,n],color='b')

plt.plot(death['China'],color='r')

plt.plot()