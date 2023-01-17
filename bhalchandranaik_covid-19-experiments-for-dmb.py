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
dataset = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')

country_region = dataset['Country/Region'].to_numpy()

print(dataset.columns)
cr_unique = np.unique(country_region)

print(cr_unique)
cases_progression = {}

country = 'China'

china_ps = dataset[dataset['Country/Region'] == country]['Province/State']

states = np.unique(china_ps)

# print(states)



# for state in states:

#     cases_progression[state] = dataset[dataset['Country/Region'] == country][dataset['Province/State'] == state]['ConfirmedCases'].to_numpy()



import matplotlib.pyplot as plt



# x = np.linspace(0.1, 1.0, 63)

# # plt.plot(140 * (1 - (1/x) + 8.5))

# fig = plt.figure(figsize=(20, 14))

# for state in states:

#     plt.plot(cases_progression[state] / np.max(cases_progression[state]), color=(1, 0, 0, np.min([1.0, np.max(cases_progression[state]) / 4000])))

# plt.legend(states)

# plt.show()
# cases_progression = {}

# china_ps = dataset[dataset['Country/Region'] == 'China']['Province/State']

# states = np.unique(china_ps)



fig = plt.figure(figsize=(20, 20))

all_states = []



# for state in states:

#     cases_progression[state] = dataset[dataset['Country/Region'] == 'China'][dataset['Province/State'] == state]['ConfirmedCases'].to_numpy()

    

# for state in states:

#     plt.plot(cases_progression[state] / np.max(cases_progression[state]), color=(1, 0, 0, np.min([1.0, np.max(cases_progression[state]) / 4000])))

# all_states = all_states + states.tolist()



cases_progression = {}

us_ps = dataset[dataset['Country/Region'] == country]['Province/State']

states = np.unique(us_ps)

print(states)



for state in states:

    cases_progression[state] = dataset[dataset['Country/Region'] == country][dataset['Province/State'] == state]['ConfirmedCases'].to_numpy()



for state in states:

    plt.plot(cases_progression[state] / np.max(cases_progression[state]), color=(0, 0, 1, np.min([1.0, np.max(cases_progression[state]) / 4000])))



all_states = all_states + states.tolist()

plt.legend(all_states)

plt.show()
from scipy.optimize import minimize

from numpy import linalg as LA

import matplotlib.pylab as plt



region = 'Hunan'

# N = 90000000

N = 67370000 #Hunan

# N = 58500000

# N = 1300000000

# N = 5696000

cases_progression['Hunan'] = dataset[dataset['Country/Region'] == 'China'][dataset['Province/State'] == region]['ConfirmedCases'].to_numpy()

print(cases_progression['Hunan'])

def infection(S, I, N, records, beta, gamma):

#     endemic_inf = (1-gamma/beta) * N

#     V = endemic_inf/I - 1

#     for t in range (0, 63):

#         I =  endemic_inf/ (1 + V*np.exp((beta-gamma)*t))

#         sus.append(S - I)

#         inf.append(I)

    sus, inf = [], []

    for t in range (0, records):

        S = S - (beta*S*I/N) + gamma * I

        I = I + (beta*S*I/N) - gamma * I



        sus.append(S)

        inf.append(I)

    return inf



def rss_infection_forecast(params):

    cases = cases_progression[region] != 0

    occurrance = cases_progression[region][cases]

    I = occurrance[0]

    

    S = N - I

    actual = cases_progression[region]

    beta, gamma = params[0], params[1]

    sus, inf = [], []

    for t in range (0, len(occurrance)):

        S = S - (beta*S*I/N) + gamma * I

        I = I + (beta*S*I/N) - gamma * I



        sus.append(S)

        inf.append(I)

    return LA.norm(np.array(inf) - occurrance, ord=2)



params = np.array([0.1, 0.1])



res = minimize(rss_infection_forecast, params, method='nelder-mead', options={'xatol': 1e-8, 'disp': True})



cases = cases_progression[region] != 0

occurrance = cases_progression[region][cases]



I = occurrance[0]

S = N - I

print(res.x)

print(len(occurrance))

inf = infection(S, I, N, 60, res.x[0], res.x[1])

plt.plot(cases_progression['Hunan'])

plt.plot(inf, 'r')

# plt.plot(occurrance, 'b')

# plt.plot(range(len(occurrance)-1, len(occurrance)+4), [694, 834, 918, 1024, 1251], 'o')

# # plt.plot(range(16, 20), [1430, 1734, 2061, 2307], 'o')

plt.show()