import pandas as pd

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
%matplotlib inline



import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from sklearn import linear_model

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score



plt.rc('figure', figsize=(12, 8.0))
#The only features you'll use as input into the model are the previous day's closing price 

#and a three day trend value. The trend value can only take on two values, either -1 or +1. 

#If the AAPL stock price has increased over any two of the previous three days then the trend will be +1. Otherwise, the trend value will be -1.

df=pd.read_csv('/kaggle/input/week4data/1NewAAPL10Y.csv')
print(type(df))

df.dropna(inplace=True)

df.head()
df.plot(x='date', y='close');
start_date = '2018-06-01'

end_date = '2018-07-31'



plt.plot(

    'date', 'close', 'k--',

    data = (

        df.loc[pd.to_datetime(df.date).between(start_date, end_date)]

    )

)



plt.scatter(

    'date', 'close', color='b', label='pos trend', 

    data = (

        df.loc[df.trend_3_day == 1 & pd.to_datetime(df.date).between(start_date, end_date)]

    )

)



plt.scatter(

    'date', 'close', color='r', label='neg trend',

    data = (

        df.loc[(df.trend_3_day == -1) & pd.to_datetime(df.date).between(start_date, end_date)]

    )

)



plt.legend()

plt.xticks(rotation = 90);
df.shape
features = ['day_prev_close', 'trend_3_day']

target = 'close'



X_train, X_test = df.loc[:2000, features], df.loc[2000:, features]

y_train, y_test = df.loc[:2000, target], df.loc[2000:, target]
# Create linear regression object

regr = linear_model.LinearRegression(fit_intercept=False)
# Train the model using the training set

regr.fit(X_train, y_train)
# Make predictions using the testing set

y_pred = regr.predict(X_test)
# The mean squared error

print('Root Mean Squared Error: {0:.2f}'.format(np.sqrt(mean_squared_error(y_test, y_pred))))



# Explained variance score: 1 is perfect prediction

print('Variance Score: {0:.2f}'.format(r2_score(y_test, y_pred)))
plt.scatter(y_test, y_pred)

plt.plot([140, 240], [140, 240], 'r--', label='perfect fit')

plt.xlabel('Actual')

plt.ylabel('Predicted')

plt.legend();
print('Root Mean Squared Error: {0:.2f}'.format(np.sqrt(mean_squared_error(y_test, X_test.day_prev_close))))