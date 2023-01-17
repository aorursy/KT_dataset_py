# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline

from datetime import datetime



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv', parse_dates=['Date'])

df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv', parse_dates=['Date'])
df_train.head()
df_train.info()
df_test.head()
#df_train['Date'] = pd.to_datetime(df_train['Date'])
df_train['Date'].min(), df_train['Date'].max()
df_train.info()
# plot the trends by country

for country in df_train['Country/Region'].unique():

    df_train_country = df_train[df_train['Country/Region'] == country]

    # if there are no provinces, then plot for the country

    if df_train_country['Province/State'].isna().unique() == True:

        plot_title = country + ' Confirmed Cases'

        # plot

        #ax, fig = plt.subplots(figsize=(4*6.4, 4*4.8))

        ax = plt.gca()

        xaxis = df_train_country['Date'].tolist()

        yaxis = df_train_country['ConfirmedCases']

        

        # changing the date format on the plot

        # Convert datetime objects to Matplotlib dates

        xaxis = matplotlib.dates.date2num(xaxis)

        hfmt = matplotlib.dates.DateFormatter('%m\n%d')

        ax.xaxis.set_major_formatter(hfmt)

        

        plt.plot(xaxis, yaxis)

        plt.title(plot_title)

        plt.tight_layout()

        plt.show()

        

        

    else:

        # plot for each of the provinces/state in each country

        state_count = len(df_train_country['Province/State'].unique())

        # split the plot into 4 columns

        num_rows = state_count / 4 + 1

        num_cols = 4

        index =1

        

        #fig = plt.figure(figsize=(20, 10))

        fig =plt.figure(figsize = (4*6.4,num_rows*4.8))

        

        for state in df_train_country['Province/State'].unique():

            df_train_state = df_train_country[df_train_country['Province/State'] == state]

            plot_title = country + '  '+ state + ' Confirmed Cases'

            

            # plot

            ax = fig.add_subplot(num_rows, num_cols, index)

            x_axis = df_train_state['Date'].tolist()

            y_axis = df_train_state['ConfirmedCases']

            

            # change the date format

            x_axis = matplotlib.dates.date2num(x_axis)

            hfmt = matplotlib.dates.DateFormatter('%m\n%d')

            ax.xaxis.set_major_formatter(hfmt)

            

            ax.plot(x_axis, y_axis)

            ax.set_xlabel('Date')

            ax.set_ylabel('Confirmed Cases')

            ax.set_title(plot_title)

            fig.tight_layout()

            index+=1

        plt.show()

            

    
df_train.head()

x_train = df_train[[]]
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline

#from sklearn import linear_model



# Prediction for Confirmed Cases



for country in df_train['Country/Region'].unique():

    print('Model for country: {}'.format(country))

    df_train_country = df_train[df_train['Country/Region'] == country]

    df_test_country = df_test[df_test['Country/Region'] == country]

    

    if df_train_country['Province/State'].isna().unique() == True:

        x_train = np.array(range(len(df_train_country))).reshape((-1, 1))

        y_train = df_train_country['ConfirmedCases']

        

        model= Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression(fit_intercept=False))])

        model.fit(x_train, y_train)

        

        x_test = np.array(range(len(df_test_country))).reshape((-1, 1))

        prediction = model.predict(x_test)

        

        # Add new column for ConfirmedCases in df_test

        df_test.loc[df_test['Country/Region'] == country, 'ConfirmedCases'] = prediction

        

        

    else:

        for state in df_train_country['Province/State'].unique():

            df_train_state = df_train_country[df_train_country['Province/State'] == state]

            df_test_state = df_test_country[df_test_country['Province/State'] == state]

            

            

            x_train = np.array(range(len(df_train_state))).reshape(-1, 1)

            y_train = df_train_state['ConfirmedCases']

            

            model= Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression(fit_intercept=False))])

            model.fit(x_train, y_train)

            

            x_test = np.array(range(len(df_test_state))).reshape((-1, 1))

            prediction = model.predict(x_test)

            

            # Add new column for ConfirmedCases in df_test

            df_test.loc[(df_test['Country/Region'] == country) & (df_test['Province/State'] == state), 'ConfirmedCases'] = prediction

             

        

                          

    
for country in df_train['Country/Region'].unique():

    print('Model for country: {}'.format(country))

    df_train_country = df_train[df_train['Country/Region'] == country]

    df_test_country = df_test[df_test['Country/Region'] == country]

    

    if df_train_country['Province/State'].isna().unique() == True:

        x_train = np.array(range(len(df_train_country))).reshape((-1, 1))

        y_train = df_train_country['Fatalities']

        

        model= Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression(fit_intercept=False))])

        model.fit(x_train, y_train)

        

        x_test = np.array(range(len(df_test_country))).reshape((-1, 1))

        prediction = model.predict(x_test)

        

        # Add new column for Fatalities in df_test

        df_test.loc[df_test['Country/Region'] == country, 'Fatalities'] = prediction

        

    else:

        for state in df_train_country['Province/State'].unique():

            df_train_state = df_train_country[df_train_country['Province/State'] == state]

            df_test_state = df_test_country[df_test_country['Province/State'] == state]

            

            

            x_train = np.array(range(len(df_train_state))).reshape(-1, 1)

            y_train = df_train_state['Fatalities']

            

            model= Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression(fit_intercept=False))])

            model.fit(x_train, y_train)

            

            x_test = np.array(range(len(df_test_state))).reshape((-1, 1))

            prediction = model.predict(x_test)

            

            # Add new column for Fatalities in df_test

            df_test.loc[(df_test['Country/Region'] == country) & (df_test['Province/State'] == state), 'Fatalities'] = prediction
# df_submit = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/submission.csv')

# df_submit['ConfirmedCases'] = df_test['ConfirmedCases'].astype('int')

# df_submit['Fatalities'] = df_test['Fatalities'].astype('int')

# df_submit.to_csv('submission.csv', index=False)
df_test.head()
df_train.info()
df_train['Lat'] = df_train['Lat'].fillna(0)

df_train['Long'] = df_train['Long'].fillna(0)



df_test['Lat'] = df_test['Lat'].fillna(0)

df_test['Long'] = df_test['Long'].fillna(0)
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion='entropy')



x_train = df_train[['Lat', 'Long']]

y_train = df_train['ConfirmedCases']



x_test = df_test[['Lat', 'Long']]



df_submit_temp = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/submission.csv')

df_submit_temp['ConfirmedCases'] = dtc.fit(x_train, y_train).predict(x_test)
x_train = df_train[['Lat', 'Long']]

y_train = df_train['Fatalities']



x_test = df_test[['Lat', 'Long']]



df_submit_temp['Fatalities'] = dtc.fit(x_train, y_train).predict(x_test)
df_submit_temp.to_csv('submission.csv', index=False)