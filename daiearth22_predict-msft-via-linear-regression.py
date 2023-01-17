'''Predict stock market prices, make billions.'''



# pylint: disable=invalid-name



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



# load data in numpy array

STOCK_SYMBOL = 'MSFT'

ALL_PRICES = pd.read_csv('../input/prices.csv')

STOCK_PRICES = np.array(ALL_PRICES[ALL_PRICES['symbol'] == STOCK_SYMBOL])



# csv column indexes

DATE_COL = 0

SYMBOL_COL = 1

OPEN_COL = 2

CLOSE_COL = 3

LOW_COL = 4

HIGH_COL = 5

VOLUME_COL = 6



# hyper-parameters

WINDOW_SIZE = 20

TRAINING_RATIO = 0.8
def get_r_squared(actuals, predicted):

    '''Calculate r_squared'''

    d1 = actuals - predicted

    d2 = actuals - actuals.mean()

    r_2 = 1 - d1.dot(d1) / d2.dot(d2)

    return r_2





def convert_numpy_dates_to_panda(numpy_dates):

    '''Convert numpy dates to pandas dates'''

    pd_dates = []

    for date in numpy_dates.flatten():

        pd_dates.append(pd.Timestamp(date))

    return pd_dates
# X is matrix of features and bias term

X = np.array(

    STOCK_PRICES[WINDOW_SIZE:, [OPEN_COL, LOW_COL, HIGH_COL, VOLUME_COL]],

    dtype='float'

)

X = np.concatenate((X, np.ones((len(X), 1))), axis=1)

num_orig_cols = X.shape[1]





# Y is matrix of actual output values

Y = np.array(

    STOCK_PRICES[WINDOW_SIZE:, CLOSE_COL],

    dtype='float'

)





# Dates are not features but we want to save them for plotting later

dates = np.array(

    STOCK_PRICES[WINDOW_SIZE:, [DATE_COL]],

    dtype='datetime64'

)
# Add previous closing prices to X for 'Rolling Window Linear Regression'

X = np.concatenate(

    (X, np.zeros((len(X), WINDOW_SIZE))),

    axis=1

)

for row in range(len(X)):

    for day in range(1, WINDOW_SIZE + 1):

        col_offset = num_orig_cols - 1 + day

        row_offset = WINDOW_SIZE + row - day

        X[row, col_offset] = STOCK_PRICES[row_offset, CLOSE_COL]



assert X.shape[1] == (WINDOW_SIZE + num_orig_cols)

# pd.DataFrame(X).to_csv('X.csv')

# pd.DataFrame(Y).to_csv('Y.csv')
# Create training and test sets

train_indexes = np.random.choice(

    len(X),

    round(len(X) * TRAINING_RATIO),

    replace=False

)

train_indexes.sort()

train_indexes.tolist()



test_indexes = list(range(len(X)))

for value in train_indexes:

    test_indexes.remove(value)



assert len(train_indexes) + len(test_indexes) == len(X)

for i, value in enumerate(train_indexes):

    assert value not in test_indexes



X_train = X[train_indexes]

Y_train = Y[train_indexes]

X_test = X[test_indexes]

Y_test = Y[test_indexes]
# Solve for w (weights) on training data

w = np.linalg.solve(X_train.T.dot(X_train), X_train.T.dot(Y_train))

Y_train_hat = X_train.dot(w)

train_r_2 = get_r_squared(Y_train, Y_train_hat)

print('r_squared of training set is:', train_r_2)



train_dates = convert_numpy_dates_to_panda(dates[train_indexes])

plt.title('Training set')

plt.scatter(train_dates, Y_train)

plt.plot(train_dates, Y_train_hat, color='red')

plt.show()
# Use w from training data to predict values in test data

Y_test_hat = X_test.dot(w)

test_r_2 = get_r_squared(Y_test, Y_test_hat)

print('r_squared of test set is:', test_r_2)



test_dates = convert_numpy_dates_to_panda(dates[test_indexes])

plt.title('Testing set')

plt.scatter(test_dates, Y_test)

plt.plot(test_dates, Y_test_hat, color='red')

plt.show()
