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
import matplotlib.pyplot as plt

import scipy.optimize as opt
data = pd.read_csv("../input/corona-virus-report/covid_19_clean_complete.csv")

data['Date']= pd.to_datetime(data['Date'])
data.tail()
#Country = "Italy"

#Country = "Mainland China"

Country = "Japan"



data_nc = data[data["Country/Region"]==Country][{"Date", "Confirmed", "Deaths", "Recovered"}]



data_aggregated = data_nc.resample('D', on='Date')['Confirmed'].sum()

data_current_cases = data_nc.resample('D', on='Date')['Confirmed'].sum()-data_nc.resample('D', on='Date')['Deaths'].sum()-data_nc.resample('D', on='Date')['Recovered'].sum()



data_aggregated.plot(title="Cases", marker="o")

data_current_cases.plot(title="Cases", marker="o")
data_aggregated_diff = data_aggregated.diff()

data_aggregated_diff.plot(title="Difference", marker="o")
np_data = data_aggregated.to_numpy()



#np_data = np_data[:30]



#padding = np.zeros(30)



#np_data = np.concatenate((padding, np_data))



X = np.arange(np_data.shape[0])

Y = np_data/np_data.max()
def logist(x, a, b, c, d):

    return a / (1. + np.exp(-c * (x - d))) + b
fig=plt.figure() 



extrapolate_x = np.arange(X.shape[0]+30)



for i in [1.2,1.,0.8]:

    np_data = data_aggregated.to_numpy()

    

    #np_data = np_data[:np_data.size-7]



    #padding = np.zeros(20)



    #np_data = np.concatenate((padding, np_data))



    X = np.arange(np_data.shape[0])

    Y = np_data/np_data.max()

    

    (a_, b_, c_, d_), cov = opt.curve_fit(logist, X, Y, maxfev=1000000)

    y_fit = logist(X, a_, b_, c_, d_)



    

    ax = fig.add_subplot(111)









    y_fit = logist(extrapolate_x, a_, b_, c_, d_) * i

    ax.plot(extrapolate_x, y_fit*np_data.max(), '-', label='Error '+str(i))



    inflection = logist( d_, a_, b_, c_, d_)

    ax.plot( d_, inflection*np_data.max(), 'o', color="red")





    

    

ax.plot(X, Y*np_data.max(), '.', label="Total infected")

ax.plot(X, data_current_cases, '.', label="Currently infected")

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    

plt.title(Country)

plt.xlabel('Time (days)')

plt.ylabel('Infected')

plt.show()
(a_, b_, c_, d_)