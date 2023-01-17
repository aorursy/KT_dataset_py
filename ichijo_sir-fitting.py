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
pd.set_option('display.max_columns', 100)

pd.set_option('display.width', 200)
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/test.csv')

print(train.head())

print(test.head())

train_jp = train[train['Country_Region']=='Japan'].copy()

test_jp = test[test['Country_Region']=='Japan'].copy()

train_jp.drop(['County', 'Province_State'], axis=1,  inplace=True)

test_jp.drop(['County', 'Province_State'], axis=1,  inplace=True)

print(train_jp.info())

print(test_jp.info())
train_jp['Date'] = pd.to_datetime(train_jp.Date)

train_jp['dayofyear'] = train_jp.Date.dt.dayofyear

train_jp_c = train_jp[train_jp.Target=='ConfirmedCases']

train_jp_f = train_jp[train_jp.Target=='Fatalities']

train_jp_c.tail()
# train_jp_c = train_jp_c[:95]

train_jp_c.tail()
test_jp['Date'] = pd.to_datetime(test_jp.Date)

test_jp['dayofyear'] = test_jp.Date.dt.dayofyear

test_jp_c = test_jp[test_jp.Target=='ConfirmedCases']

test_jp_f = test_jp[test_jp.Target=='Fatalities']

test_jp_c.head()
# The 3/4 of population might not be infected finally.

population = train_jp.Population.iloc[0]/4

# b = 1.5 # infecters per week

# c = 0.5 # recover ratio

# d = 0.1 # death ratio

t_arr = np.array(train_jp_c.dayofyear)

y_arr = np.array(train_jp_c.TargetValue)

test_t_arr = np.array(test_jp_c.dayofyear)



# Start from the data has more than 10 infects.

start_day = list(y_arr>=10).index(True)

print(start_day)



t_arr = t_arr[start_day:]

y_arr = y_arr[start_day:]

t_arr_first = t_arr[0]

t_arr -= t_arr_first

test_t_arr -= t_arr_first

# clensing

for i,n in enumerate(y_arr):

    if n > 0:

        if n > 1000:

            y_arr[i] = (y_arr[i-1] + y_arr[i+1]) / 2

        else:

            continue

    else:

        y_arr[i] = (y_arr[i-1] + y_arr[i+1]) / 2
from scipy import integrate, optimize





susceptible_0 = population - y_arr[0]

infected_0 = y_arr[0]



def sir(y,t,b,c,d):

    susceptible = -b * y[0] * y[1] / susceptible_0

    recovered = c * y[1]

    fatarities = d * y[1]

    infected = -(susceptible + recovered + fatarities)

    return susceptible,infected,recovered,fatarities



def inf_odeint(t,b,c,d):

    return integrate.odeint(sir,(susceptible_0,infected_0,0,0),t,args=(b,c,d))[:,1]



popt, pcov = optimize.curve_fit(inf_odeint, t_arr, y_arr)
# fitted = inf_odeint(np.append(t_arr,test_t_arr), *popt)

fitted = inf_odeint(t_arr, *popt)

import matplotlib.pyplot as plt



plt.plot(t_arr, y_arr, 'bo')

# plt.plot(np.append(t_arr,test_t_arr), fitted)

plt.plot(t_arr, fitted)



plt.title("Fit of infected model")

plt.ylabel("Population infected")

plt.xlabel("Days")

plt.show()
