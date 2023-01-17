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
import collections



dirname = '/kaggle/input/covid19-global-forecasting-week-4/'

trainname = 'train.csv'

trainpath = dirname + trainname

traindata_raw = pd.read_csv(trainpath)

# print(traindata)

traindata = collections.defaultdict(lambda :collections.defaultdict(list))

traindata_country = collections.defaultdict(lambda :collections.defaultdict(lambda :[0,0]))

traindata_state = collections.defaultdict(lambda :collections.defaultdict(lambda :[0,0]))

traindata_country_onlymainland = collections.defaultdict(lambda :collections.defaultdict(lambda :[0,0]))

for i in range(len(traindata_raw)):

    country = traindata_raw['Country_Region'][i]

    state = traindata_raw['Province_State'][i]

    date = traindata_raw['Date'][i]

    cases = traindata_raw['ConfirmedCases'][i]

    fata = traindata_raw['Fatalities'][i]

    # State daily Perspective (single row for one state each day)

    traindata[country][date].append((state, cases, fata))

    # Country daily Perspective (single row for the whole country each day)

    traindata_country[country][date][0] += cases

    traindata_country[country][date][1] += fata

    # Country mainland daily Perspective (single row for the whole country each day)

    if state != state: #NAN

        traindata_country_onlymainland[country][date][0] += cases

        traindata_country_onlymainland[country][date][1] += fata

    # State daily Perspective (single row for the whole state each day)

    state_ = country + ':' + state if state == state else country + ':'

    traindata_state[state_][date][0] += cases

    traindata_state[state_][date][1] += fata

        
# Easy Curve Fitting

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt



predictdata_state = collections.defaultdict(lambda :[[],[]])

for state in traindata_state:

    state_history = [[],[]]

    for date in traindata_state[state]:

        state_history[0].append(traindata_state[state][date][0])

        state_history[1].append(traindata_state[state][date][1])

    i = 0

    for perspective in state_history:

        perspective_train = [i for i in perspective[:] if i!=0]

        perspective = [i for i in perspective if i!=0]

        x_all = np.array(range(1, len(perspective) + 31))#np.linspace(1, 1.3, len(perspective)+3)#.reshape(-1,1)

        x = x_all[:len(perspective_train)]

        y = np.array(perspective_train)





        if len(perspective_train) < 3:

            if not perspective_train:

                predict_y = np.zeros(len(x_all[len(perspective_train):]))

            else:    

                predict_y = np.array([perspective_train[-1]]*(len(x_all[len(perspective_train):])))

            predictdata_state[state][i] = predict_y

            i = (i+1) % 2

            continue



        if True:

            def funmi(x, a, b, c):

                return (x/a)**b + c

            

            def funexp(x,a,b,c):

                return (a**(x + b))* c

            

            def funline(x, a):

                return a*x



            def funsigmoid(x, a, b, c):

                # a is the final case number(scale factor of y), determined by population, spread, medical resources etc.

                # b is the potential length of days from start to the day with maximum daily increase

                # c is the spread speed, c*5 days to reach the 90% of the final cases from the day with maximum daily increase(scale factor of x)

                s = a / (1 + np.exp(- (x - b) / c))

                return s

            

            

            p0 = [max(perspective_train)*2, max(x), 5]

            if y[-1] > 10:

                popt, pcov = curve_fit(funsigmoid, x, y, maxfev = 50000, p0=p0)

            else:

                popt, pcov = curve_fit(funline, x, y, maxfev = 50000)

#             print('State:', state)

#             print('Y:',perspective_train)

#             print('Parameters:', popt[0], popt[1], popt[2])



            predict_x = x_all[len(perspective_train):]

            if y[-1] > 10:

                severe_coefficient = 1.1 if 'China' not in state else 1.0

                predict_y = [funsigmoid(i, *popt) * severe_coefficient for i in predict_x]

            else:

                predict_y = [funline(i, *popt) for i in predict_x]

            if predict_y[0] < perspective_train[-1]:

                predict_y = [i + (perspective_train[-1] - predict_y[0]) for i in predict_y]

#             print(x[-1], funsigmoid(x[-1], *popt), y[-1])

#             print(predict_x[0], predict_y[0])

            predict_y = [round(i) for i in predict_y]

#             print('Prediction:',predict_y)

            predictdata_state[state][i] = predict_y

#             plt.plot(x,y)

#             plt.plot(x, funsigmoid(x, *popt))

#             plt.show()



        i = (i+1) % 2

# print(predictdata_state)
# Write Result

testname = 'test.csv'

testpath = dirname + testname

testdata = pd.read_csv(testpath)

submname = 'submission.csv'

submpath = dirname + submname

submdata = pd.read_csv(submpath)

res = []

ind = 0

found = False

for i in range(len(testdata)):

    forecastid = testdata['ForecastId'][i]

    state = testdata['Province_State'][i]

    country = testdata['Country_Region'][i]

    date = testdata['Date'][i]

    state_ = country + ':' + state if state == state else country + ':'

#     if date in traindata_state[state_]:

#         res.append(traindata_state[state_][date])

#         ind = 0

#     else:

#         res.append([int(predictdata_country[state_][0][ind]), int(predictdata_country[state_][1][ind])])

#         ind += 1

    if date == '2020-04-02':

        found = True

    if date == '2020-04-15':

        found = False

        ind = 0

    if found:

        res.append(traindata_state[state_][date])

    else:

#         print([int(predictdata_state[state_][0][ind]), int(predictdata_state[state_][1][ind])])

        res.append([int(predictdata_state[state_][0][ind]), int(predictdata_state[state_][1][ind])])

        ind += 1

subm_cases = [i[0] for i in res]

subm_fata = [i[1] for i in res]

submdata['ConfirmedCases'] = subm_cases

submdata['Fatalities'] = subm_fata

submdata.to_csv('submission.csv', index = False)