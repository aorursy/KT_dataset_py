# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import TimeSeriesSplit

from sklearn.metrics import mean_squared_log_error

from sklearn.compose import TransformedTargetRegressor

from sklearn.linear_model import LinearRegression,  Ridge

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import make_pipeline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_train.csv')

train.head()
# Clean the data. only look at the data from the first confirmed case

train = train[train.ConfirmedCases>0]

train
whole_world_data = pd.read_csv('/kaggle//input/covid19-global-forecasting-week-1/train.csv')

us_data = whole_world_data.loc[whole_world_data['Country/Region'] == 'US']

us_data
possible_states = us_data['Province/State'].unique()[(us_data.groupby('Province/State').max().ConfirmedCases>144 ) &  (us_data.groupby('Province/State').min().ConfirmedCases<6)]

possible_states
COLUMNS = ['ConfirmedCases', 'Fatalities', 'Province/State']

possible_starts = pd.DataFrame(columns=COLUMNS)

for country in possible_states:

    possible_starts = pd.concat([possible_starts, us_data[(us_data['Province/State'] == country) & (us_data['ConfirmedCases']<144) & (us_data['ConfirmedCases']>0)][COLUMNS]])



possible_starts
# First, find which start is the best for an exponential pattern, and what is the lowest cross validation we get with it

def test_exponential_accuraccy(y):

    accuraccy = 0

    X = np.asarray(list(range(len(y)))) +1

    tscv = TimeSeriesSplit(len(X)-1)

    for train_index, test_index in tscv.split(X):

        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]



        # Transform the data with a log function and after prediction, apply the exp() function.

        regressor = TransformedTargetRegressor(regressor=LinearRegression(),

                                                         func=np.log1p,

                                                         inverse_func=np.expm1)

        regressor.fit(np.array(X_train).reshape(-1,1), np.array(y_train).reshape(-1,1))

        y_pred = regressor.predict(np.array(X_test).reshape(-1,1))

        accuraccy += mean_squared_log_error(y_pred, y_test)

        

        

    

    return accuraccy
# For every state, compute its sliding window error.

best_accuraccy = 100

best_start_state = ''

for state in possible_starts['Province/State'].unique():

    possible_train_data = pd.concat([possible_starts[possible_starts['Province/State']==state], train])

    

    state_accuracy = test_exponential_accuraccy(possible_train_data['ConfirmedCases'].values) 

    if  state_accuracy < best_accuraccy:

        best_start_state = state

        best_accuraccy = state_accuracy

                                

print(best_start_state, best_accuraccy)
# Now, lets find which start is the best for a polynomial pattern, what is the degree of the polynom,

# and what is the lowest cross validation we get with it

def test_polynomial_accuraccy(y, degree):

    accuraccy = 0

    X = np.asarray(list(range(len(y)))) +1

    tscv = TimeSeriesSplit(len(X)-1)

    for train_index, test_index in tscv.split(X):

        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]

        

        # Fit the data with a polynom of the given degree.

        model = make_pipeline(PolynomialFeatures(degree), Ridge())

        model.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))

        y_pred = model.predict(X_test.reshape(-1,1))

#         print(type(y_pred))

        if y_pred < 0: 

            y_pred = np.array([0]) # this case can happen in the beggining of the sliding window, and a high degree polynom. y_pred=0 is a big enough error

        accuraccy += mean_squared_log_error(y_pred, y_test)

        

        

    

    return accuraccy
# For every state and polynom degree, compute its sliding window error

# What we are going to do is look at a few options and decide where we get a low enough error, and not over generalize polynom.



degrees = [2,3,4,5,6]

lowest_errors = pd.DataFrame(columns=['accuraccy', 'state', 'degree'])

for degree in degrees:

    best_accuraccy = 100

    best_start_state = ''

    for state in possible_starts['Province/State'].unique():

        possible_train_data = pd.concat([possible_starts[possible_starts['Province/State']==state], train])

    

        state_accuracy = test_polynomial_accuraccy(possible_train_data['ConfirmedCases'].values, degree)

        if  state_accuracy < best_accuraccy:

            best_start_state = state

            best_accuraccy = state_accuracy

    

    lowest_errors = lowest_errors.append(pd.DataFrame({'accuraccy': [best_accuraccy], 'state': [best_start_state], 'degree': [degree]}))
lowest_errors
possible_starts[possible_starts['Province/State'] == 'Massachusetts']
train = pd.concat([possible_starts[possible_starts['Province/State'] == 'Massachusetts'], train])[['ConfirmedCases', 'Fatalities']]

train = train.loc[train.ConfirmedCases>0]
train.reset_index()
days_to_predict = 43 # Change to 29

public_leader_board_first_column=7 # Change to 26

model = make_pipeline(PolynomialFeatures(3), Ridge())



#ConfirmedCases predictions

X_train = np.array(range(len(train))) + 1

X_test = np.array(range(public_leader_board_first_column,public_leader_board_first_column+days_to_predict)) + 1

y_train = train.ConfirmedCases.values

model.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))

confirmed_cases_predictions = model.predict(X_test.reshape(-1,1))

confirmed_cases_predictions = list(map(lambda x: x[0], confirmed_cases_predictions.tolist()))



#Fatalities predictions

y_train = train.Fatalities.values

model.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))

fatalities_predictions = model.predict(X_test.reshape(-1,1))

fatalities_predictions = list(map(lambda x: x[0], fatalities_predictions.tolist()))



submissions = pd.DataFrame({'ConfirmedCases': confirmed_cases_predictions, 'Fatalities': fatalities_predictions})

submissions.to_csv('submission.csv', index=False)


