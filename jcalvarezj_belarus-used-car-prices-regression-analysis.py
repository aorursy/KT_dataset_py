# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
cars_df = pd.read_csv('/kaggle/input/belarus-used-cars-prices/cars.csv')

cars_df.isna().any()
cars_df['make_count'] = cars_df.groupby('make')['make'].transform('count')

cars_df[cars_df['make_count'] == 3]
def mark_for_deletion(series):

    return len(set(series)) == 1

    

#conteos = cars_df.groupby('make')['year'].value_counts()

#conteos

cars_df['deletion'] = cars_df.groupby('make')['year'].transform(mark_for_deletion)

deletion_rows = cars_df[cars_df['deletion']]

print(f'Deleting {len(deletion_rows)} records with only one registering production year')

cars_df.drop(deletion_rows.index, inplace = True)
import matplotlib.pyplot as plt





def plot_makes(df, xlabel = 'Year', ylabel = 'Price (USD)', title = 'Price per Year for different car makes'):

    _, ax = plt.subplots(figsize = (30,28))

    ax.set_xlabel(xlabel)

    ax.set_ylabel(ylabel)    

    ax.set_title(title)

    for make, make_df in cars_df.groupby('make'):        

        ax.scatter(make_df['year'], make_df['priceUSD'], label = make)

    ax.legend()





plot_makes(cars_df)
next_year = cars_df['year'].max() + 1

print(f'Year of prediction: {next_year}')
import statsmodels.api as sm

from statsmodels.graphics.regressionplots import abline_plot





def plot_regression_prediction(x, y, x_pred, y_pred, regression, xlabel = '', ylabel = '', title = ''):

    fig, ax = plt.subplots(figsize = (30,21))

    ax.scatter(x, y, label = 'real')

    ax.scatter(x_pred, y_pred, label = f'{make} prediction')

    ax.set_xlabel(xlabel)

    ax.set_ylabel(ylabel)

    ax.set_title(title)

    abline_plot(model_results = regression, ax = ax)

    



def obtain_make_regression_model(make):

    """

    Returns the fit OLS linear regression instance for a make

    """

    make_df = cars_df[cars_df['make'] == make]

    

    y = make_df['priceUSD']

    x = make_df['year']



    x_pred = pd.DataFrame([next_year, next_year + 1])

    return sm.OLS(y, sm.add_constant(x)).fit()





def get_regressions_dataframe(df, x_pred):

    reg_df = pd.DataFrame(columns = ['make','prediction'])

    reg_df['make'] = cars_df['make'].unique()

    reg_df.set_index('make', inplace = True)



    for make in reg_df.index:

        linear_regression = obtain_make_regression_model(make)

        reg_df.loc[make, 'prediction'] = linear_regression.predict(sm.add_constant(x_pred))[0]



    return reg_df





reg_df = get_regressions_dataframe(cars_df, [next_year, next_year + 1])



max_price_car = reg_df[reg_df['prediction'] == reg_df['prediction'].max()]



print(f'The car make with the highest expected price for {next_year} is {max_price_car.index[0]}, with a price of {max_price_car["prediction"].values[0]:.2f} USD')
max_price_car_df = cars_df[cars_df['make'] == max_price_car.index[0]]

max_price_car_df
def plot_regression_prediction(x, y, x_pred, y_pred, regression, make = ''):

    fig, ax = plt.subplots(figsize = (10,7))

    ax.scatter(x, y, label = f'{make} real')

    ax.scatter(x_pred, y_pred, label = f'{make} prediction')

    ax.set_xlabel('Year')

    ax.set_ylabel('Price (USD)')

    ax.set_title(f'Price per Year regression for cars of the {make} make')

    ax.legend()

    abline_plot(model_results = regression, ax = ax)



reg = obtain_make_regression_model(max_price_car.index[0])

    

plot_regression_prediction(max_price_car_df['year'], max_price_car_df['priceUSD'], next_year, max_price_car['prediction'], reg, make = max_price_car.index[0])