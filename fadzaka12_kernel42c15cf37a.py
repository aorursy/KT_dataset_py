# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestRegressor

from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = (10,6)

pd.set_option('max_column', 100)

pd.set_option('max_row', 200)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
## Preprocess the data

def preprocess (dataset):

    processed_dataset = dataset.copy()

    

    # Your code goes here

    

    return processed_dataset
## Split train and test

def split_train_test(dataset, end_of_training_date):

    

    # training_data =

    # testing_data = 

    

    return training_data, testing_data
## Split label and predictor

def split_label_and_predictor(train_or_test_data):

    

    # x_data = 

    # y_data = 

    

    return x_data, y_data
def fit(x_train, y_train):

    regr = RandomForestRegressor()

    regr.fit(x_train, y_train)

    return regr
def predict(est, x_test):

    y_pred = est.predict(x_test)

    return y_pred
df = pd.read_csv('/kaggle/input/uisummerschool/Online_sales.csv')

df['Date'] = df['Date'].astype(str)

df['Date'] = pd.to_datetime(df['Date'])

df.head()
df_rev = df.groupby('Date')['Revenue'].sum().reset_index(name='Revenue')

add_data = [['2017-12-01', 0], ['2017-12-02', 0], ['2017-12-03', 0],

            ['2017-12-04', 0], ['2017-12-05', 0], ['2017-12-06', 0],

            ['2017-12-07', 0], ['2017-12-08', 0], ['2017-12-09', 0],

            ['2017-12-10', 0], ['2017-12-11', 0], ['2017-12-10', 0],

            ['2017-12-13', 0], ['2017-12-14', 0]

           ] 

  

# Create the pandas DataFrame 

add_data_df = pd.DataFrame(add_data, columns = ['Date', 'Revenue']) 

add_data_df['Date'] = add_data_df['Date'].astype(str)

add_data_df['Date'] = pd.to_datetime(add_data_df['Date'])



df_rev = df_rev.append(add_data_df)

df_rev.head()
## Preprocess

daily_online_revenue = preprocess(df_rev).set_index('Date')
## Split

training_data, test_data = split_train_test(daily_online_revenue,"2017-11-30")

X_train, y_train = split_label_and_predictor(training_data)

X_test, y_test = split_label_and_predictor(test_data)
# Fit the model

model = fit(X_train, y_train)
# Predict the model

a = list(daily_online_revenue.iloc[-5:, :]['Revenue'])

for i in range(0, 5):

    temp = a[-5:]

    y = predict(model, [temp])

    a.append(y[0])
# Save the result to CSV



# Your code goes here
