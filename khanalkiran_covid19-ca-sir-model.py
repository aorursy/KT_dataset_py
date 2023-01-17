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
from scipy.integrate import odeint

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

from matplotlib.dates import DateFormatter, DayLocator, WeekdayLocator

import datetime as dt

from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU

from matplotlib import gridspec

from matplotlib import dates

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,

                               AutoMinorLocator)



train_data = pd.read_csv("/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_train.csv")

test_data = pd.read_csv("/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_test.csv")
train_data.head()
test_data.head()
train_data.shape, test_data.shape
fig, (ax1, ax2) = plt.subplots(1,2, figsize = (18,8))

fig.suptitle('Number of Confirmed Cases and Fatalities in CA')

plt .xticks(np.arange(0, 60,  step = 10)) 

#Left plot

ax1.plot(train_data['Date'], train_data['ConfirmedCases'], color = 'purple', marker = 'o',linewidth = 1)

ax1.set(xlabel = 'Date',

        ylabel = 'Number of ConfirmedCases in CA')

ax1.set_xticks(np.arange(0, 60,  step = 12))

ax1.grid()

#Right plot



ax2.plot(train_data['Date'], train_data['Fatalities'], color = 'orange', marker = 'o', linewidth = 1)

ax2.set(xlabel = 'Date',

        ylabel = 'Number of Fatalities in CA')

ax2.set_xticks(np.arange(0, 60,  step = 12))

ax2.grid()



plt.rcParams["font.family"] = "Times New Roman"

plt.rcParams["font.size"] = "12"



plt.show()
train_initial = train_data[48:]

train_initial = train_initial.reset_index()

y1_train= train_initial['ConfirmedCases']

y2_train = train_initial['Fatalities']
#SIR Model 

def SIR_DEQ(y, time, beta, k, N):

    DS = -beta * y[0] * y[1]/N

    DI = (beta * y[0] * y[1] - k * y[1])/N

    DR = k * y[1]/N

    return [DS, DI, DR]



#initial conditions for training data

N = 39560000  #Population of California 

I0 = 144

S0 = N  # initial population of susceptible individual

R0 = 2 # initial number of fatalities 

init_state = [S0, I0, R0]



# Parameters

t0 = 0 

tmax = 15

dt = 1

# Rate of infection

beta = 0.2

# Rate of recovery

k = 1/10



time = np.arange(t0, tmax, dt)

args = (beta, k, N)



solution = odeint(SIR_DEQ, init_state, time, args)



plt.plot(time, solution[:, 1], 'g', marker = 'x', label  = 'Infected SIR model')

plt.plot(time, y1_train, 'r', marker = 'o', label  = 'Infected Data')

plt.legend(['Infected SIR Model', 'Infected Input Data'])
#initial conditions for ouput

N = 39560000  #Population of California 

I0 = 221

S0 = N  # initial population of susceptible individual

R0 = 4 # initial number of fatalities 

init_state_out = [S0, I0, R0]



time = np.arange(t0, 43, dt)

args = (beta, k, N)

solution_out = odeint(SIR_DEQ, init_state_out, time, args)

submission_file = pd.read_csv("/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_submission.csv")

submission_file['ConfirmedCases']= solution_out[:,1]

submission_file['Fatalities'] = solution_out[:,2]
submission_file.to_csv("submission.csv", index=False)