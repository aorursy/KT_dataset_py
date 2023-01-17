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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import os

from statsmodels.tsa.api import ExponentialSmoothing, Holt

import datetime

import warnings

warnings.simplefilter('ignore')



path='../input/demand-forecasting-kernels-only/' # to use when running in kaggle

#path='./input/' # for local use # comment "#" this line if running in kaggle



for dirname, dirnames, filenames in os.walk(path):

    # print path to all subdirectories first.

    for subdirname in dirnames:

        print(os.path.join(dirname, subdirname))



    # print path to all filenames.

    for filename in filenames:

        print(os.path.join(dirname, filename))
start = datetime.datetime.now() # for timming purposes

# load the data and check contents

df = pd.read_csv(path + 'train.csv', parse_dates=['date'], index_col='date')

#df.index = df.index.to_period('D') 

df.iloc[[0, 1825, 1826, 3652, 18259, 18260, 18261, -1]]
# view the dimensions of the data

df.pivot_table(index='store', columns='item', values='sales', aggfunc={'sales':'sum'}).fillna(0).astype('int')
# view basic drawing

df.pivot_table(index='store', columns='item', values='sales', 

               aggfunc={'sales':'sum'}).fillna(0).astype('int').plot(figsize=(15,15),

                title='Items sold per store', legend=True);
item = 6

store = 3

nperiods = 90

degree = 3



data = df[(df['store'] == store) & (df['item'] == item)]['sales'].to_frame()



y = data.reset_index()['sales'].to_numpy()

x = range(len(y))



weights = np.polyfit(x, y, degree)



formula = np.poly1d(weights)



future_dates = pd.date_range(data.index.max() + pd.Timedelta('1 day'), periods=nperiods)



future_x = range(len(y)+1, len(y)+nperiods+1, 1) # values for the x axis for the prediction

formula_predictions = np.polyval(formula, future_x, )

pd.DataFrame(formula_predictions, columns=['sales'])
# graphics section

plt.figure(figsize=(14, 6))

plt.plot(data.index, data.sales, 'b-', label='Actual')

plt.plot(future_dates, formula_predictions, 'ro', label='Forecast')

plt.legend(loc="best")

plt.gcf().suptitle('Forecast using numpy.polyfit')

plt.title('Forecasted sales of item {} at store {}'.format(item, store))

plt.show();

#plt.close('all') # clean up after using pyplot
nperiods = 90 # periods to forecast (using 90 as the information is daily)



store = 3

item = 6

degrees = 11



salesdf = pd.DataFrame()



for degree in range(1, degrees):

    data = df[(df['store'] == store) & (df['item'] == item)]['sales'].to_frame()



    y = data.reset_index()['sales'].to_numpy()

    x = range(len(y))



    weights = np.polyfit(x, y, degree)

    formula = np.poly1d(weights)



    future_x = range(len(y)+1, len(y)+nperiods+1, 1) # values for the x axis for the prediction

    formula_predictions = np.polyval(formula, future_x, )

    x = pd.DataFrame(formula_predictions, columns=[degree]) 

    salesdf = salesdf.append(x, ignore_index=False)



salesdf.iloc[[0,-1]]
salesdf.plot(title='Forecast for item {} at store {} using {} different degrees'.format(item, store, degrees-1), figsize=(15,6));
salesdf = pd.DataFrame(columns=['sales'])

nperiods = 90 # periods to forecast (using 90 as the information is daily)

degree = 3



for item in range (1, 51):

    for store in range (1, 11): # there are 10 stores and need to add 1 to run it 10 times

        data = df[(df['store'] == store) & (df['item'] == item)]['sales'].to_frame()



        y = data.reset_index()['sales'].to_numpy()

        x = range(len(y))



        weights = np.polyfit(x, y, degree)

        formula = np.poly1d(weights)



        future_x = range(len(y)+1, len(y)+nperiods+1, 1) # values for the x axis for the prediction

        formula_predictions = np.polyval(formula, future_x, )

        x = pd.DataFrame(formula_predictions, columns=['sales']) 

        salesdf = salesdf.append(x, ignore_index=True)

        

# Generating the submission file

salesdf.index.name = 'id'

salesdf.to_csv('store_forecast_numpy_polyfit.csv', index=True)

salesdf.iloc[[0, -1]]
salesdf[salesdf.isnull().any(axis=1)] # checking for NaN
# graphics section

plt.figure(figsize=(14, 6))

plt.plot(salesdf.reset_index().id, salesdf.sales, 'ro', label='Forecast')

plt.legend(loc="best")

plt.gcf().suptitle('Forecast using numpy.polyfit')

plt.title('looking at the formula_predictions')

plt.show();

plt.close('all') # clean up after using pyplot
end = datetime.datetime.now()

print('start:\t{}\r\nend:\t{}\r\nDelta:\t{}'.format(start, end, end - start))
salesdf2 = pd.DataFrame(columns=['sales'])

nperiods = 90 # periods to forecast (using 90 as the information is daily)

degree = 4



for item in range (1, 51):

    for store in range (1, 11): # there are 10 stores and need to add 1 to run it 10 times

        data = df[(df['store'] == store) & (df['item'] == item)]['sales'].to_frame()



        y = data.reset_index()['sales'].to_numpy()

        x = range(len(y))



        weights = np.polyfit(x, y, degree)

        formula = np.poly1d(weights)



        future_x = range(len(y)+1, len(y)+nperiods+1, 1) # values for the x axis for the prediction

        formula_predictions = np.polyval(formula, future_x, )

        x = pd.DataFrame(formula_predictions, columns=['sales']) 

        salesdf2 = salesdf2.append(x, ignore_index=True)

        

# Generating the submission file

salesdf2.index.name = 'id'

salesdf2.iloc[[0, -1]]
# graphics section

plt.figure(figsize=(15, 6))

plt.plot(salesdf.reset_index().id, salesdf.sales, 'r.', label='Forecast degree 3')

plt.plot(salesdf2.reset_index().id, salesdf2.sales, 'b.', label='Forecast degreee 4')

plt.legend(loc="best")

plt.gcf().suptitle('Forecast using numpy.polyfit')

plt.title('comparing two runs of the formula_predictions for ll items and stores')

plt.show();

plt.close('all') # clean up after using pyplot



for dirname, _, filenames in os.walk('.'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        
