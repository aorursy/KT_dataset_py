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

        print(filename)



# Any results you write to the current directory are saved as output.
# import extra libraries 

import datetime

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt



from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
def plot_predictions(state='Sweden',

                    thrs=20,

                    num_feature_days=3,

                    max_num_previous_days=10,

                    num_future_days=7):

    '''

    Args:

        state (str) - name of the State

        thrs (int) - minimum number of registered cases to model with

        num_feature_days - number of previous days ratios include to feature modeling

        max_num_previous_days - maximal number of previous days used for modeling

        num_future_days - number of future days include to prediction

    Returns:

        None

    '''

    ### import data

    df = pd.read_csv(os.path.join(dirname, 'covid_19_clean_complete.csv'))

    

    df['Date'] = df['Date'].astype(np.datetime64)

    df = df[df['Country/Region'] == state]

    df.set_index('Date', inplace=True)



    df1 = df[df['Confirmed'] >= thrs]

    cols = ['Confirmed', 'Deaths']

    df1 = df1[cols]

    df1 = df1.groupby('Date').agg(sum) # sum data from all regions



    log_transf = lambda x: np.log10(1+x)

    df1['log_confirmed'] = df1['Confirmed'].map(log_transf)

    df1['log_confirmed_ratio'] = df1['log_confirmed']-df1['log_confirmed'].shift(1)

    #df1['log_deaths'] = df1['Deaths'].map(log_transf)

    #df1['log_deaths_ratio'] = df1['log_deaths']-df1['log_deaths'].shift(1)

    df1.dropna(axis=0, inplace=True)

    df1.drop(['Confirmed', 'Deaths'], axis=1, inplace=True)

    ### set ratios from num_feature_days previous days as features

    for col in ['log_confirmed_ratio']:

        for i in range(1,num_feature_days+1):

            df1[f"{col}_{i}"] = df1[f"{col}"].shift(i)

    df1.dropna(axis=0, inplace=True)

    ### include no more than max_num_previous_days to modeling

    df1 = df1.tail(max_num_previous_days)

    cols2drop = ['log_confirmed', 'log_confirmed_ratio']

#    print(cols2drop)

    y = df1['log_confirmed_ratio']

    X = df1.drop(cols2drop, axis=1)

    y_train = y.head(df1.shape[0]-3)

    X_train = X.head(df1.shape[0]-3)

    y_val = y.tail(3)

    X_val = X.tail(3)



    lr = LinearRegression()

    lr.fit(X_train, y_train)



    y_train_pred = lr.predict(X_train)

    mae_lr_train = mean_absolute_error(y_train, y_train_pred)

    print(f"Use {X.shape[0]} days for modeling")



    y_val_pred = lr.predict(X_val)

    mae_lr_val = mean_absolute_error(y_val, y_val_pred)

    print(f"MAE train error {mae_lr_train.round(4)}, val error {mae_lr_val.round(4)}")

    

    ### refit the data using the whole dataset for making predictions

    lr.fit(X, y)



    ### predict for N days in future:

    df2 = df1[[c for c in df1.columns if c.startswith('log_confirmed')]]

    for _ in range(num_future_days):

        new_day = df2.index[-1] + datetime.timedelta(days=1)

        cols_predict = [c for c in df2.columns if c.startswith('log_confirmed_ratio_')]

        for col in ['log_confirmed_ratio']:

            for i in range(1,num_feature_days+1):

                df2.loc[new_day,f"{col}_{i}"] = df2.loc[new_day-datetime.timedelta(days=i),f"{col}"]

        df2.loc[new_day, 'log_confirmed_ratio'] = lr.predict(df2.loc[new_day, cols_predict].values.reshape(1,-1))

        df2.loc[new_day, 'log_confirmed'] = df2.loc[new_day-datetime.timedelta(days=1), 'log_confirmed'] + df2.loc[new_day, 'log_confirmed_ratio']



    df2['confirmed'] = df2['log_confirmed'].map(lambda x: int(10**x-1))

    print(f"We predict {df2.confirmed.tail(1).values[0]} registered cases for {state} by {df2.index[-1]} ")



    plt.rcParams["figure.figsize"] = (8,6)

    plt.title(state)

    plt.yscale('log')

    plt.xticks(rotation=45)

    plt.plot(df[df['Confirmed'] >= thrs][cols].groupby('Date').agg(sum).Confirmed.tail(15), label='actual cases')

    plt.plot(df2.tail(num_future_days+2)['confirmed'], label='predicted cases')

    plt.legend()

    plt.show()
for state in ['Sweden', 

              'Denmark', 

              'Netherlands', 

              'US', 

              'Italy', 

              'Germany', 

              'Spain', 

              'France', 

              'United Kingdom', 

              'Norway', 

              'Belgium', 

              'India', 

              'Russia', 

              'Canada', 

              'Australia', 

              'Thailand',

              'Ukraine'

             ]:

    print(state)

    plot_predictions(state=state)