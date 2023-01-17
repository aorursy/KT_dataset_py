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
cricket_df = pd.read_csv('/kaggle/input/cricket-chirp-vs-temperature/Cricket_chirps.csv')

cricket_df.rename(columns = {'X': 'temperature', 'Y': 'chirps'}, inplace = True)



cricket_df.isna().any()
import matplotlib.pyplot as plt





def plot_measure(x, y, xlabel = '', ylabel = '', title = ''):

    fig, ax = plt.subplots(figsize = (11,7))

    ax.scatter(x, y)

    ax.set_xlabel(xlabel)

    ax.set_ylabel(ylabel)

    ax.set_title(title)

    

    

def plot_regression_prediction(x, y, x_pred, y_pred, regression, xlabel = '', ylabel = '', title = ''):

    fig, ax = plt.subplots(figsize = (11,7))

    ax.scatter(x, y, label = 'real')

    ax.scatter(x_pred, y_pred, label = 'prediction')

    ax.set_xlabel(xlabel)

    ax.set_ylabel(ylabel)

    ax.set_title(title)

    abline_plot(model_results = regression, ax = ax)



x = cricket_df['temperature']

y = cricket_df['chirps']



plot_measure(x, y, xlabel = 'Temperature (Fahrenheit)', ylabel = 'Chirps per Second', title = 'Chirps per second vs. temperature in Fahrenheit')
lowest_temp = cricket_df['temperature'].min()

highest_temp = cricket_df['temperature'].max()



print(f'The lowest temperature in the dataset is {lowest_temp}, and the highest is {highest_temp}')
import statsmodels.api as sm

from statsmodels.graphics.regressionplots import abline_plot





linear_regression = sm.OLS(y, sm.add_constant(x)).fit()



x_right_pred = pd.DataFrame(range(int(highest_temp) + 1, int(highest_temp) + 21))



y_right_pred_raw = linear_regression.predict(sm.add_constant(x_right_pred))

y_right_pred = pd.DataFrame(y_right_pred_raw).set_index(x_right_pred[0])



plot_regression_prediction(x, y, x_right_pred, y_right_pred, linear_regression, xlabel = 'Temperature (Fahrenheit)', ylabel = 'Chirps per Second', title = 'Chirps per second vs. temperature in Fahrenheit')
b = linear_regression.params[0]

slope = linear_regression.params[1]

intercept_on_y = -b/slope

print(f'The minimum temperature for obtaining zero chirps per second is {intercept_on_y} Fahrenheit')
from statsmodels.graphics.regressionplots import abline_plot





x_left_pred = pd.DataFrame(range(int(intercept_on_y - 1), int(lowest_temp)))



y_left_pred_raw = linear_regression.predict(sm.add_constant(x_left_pred))

y_left_pred = pd.DataFrame(y_left_pred_raw).set_index(x_left_pred[0])



plot_regression_prediction(x, y, x_left_pred, y_left_pred, linear_regression, xlabel = 'Temperature (Fahrenheit)', ylabel = 'Chirps per Second', title = 'Chirps per second vs. temperature in Fahrenheit')