# This Python 3 environment comes with many helpfula analytics libraries installed

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
import matplotlib.pyplot as plt

import seaborn as sns



# Set global parameters for plotting.

plt.rc('figure', figsize=(12, 6))

sns.set(font_scale=1.2)
import warnings



warnings.filterwarnings('ignore')
DATASET_PATH = '/kaggle/input/avocado-prices-2020/avocado-updated-2020.csv'



avocado_df = pd.read_csv(DATASET_PATH, 

                         parse_dates=['date'],

                         index_col=['date'])



avocado_df
columns_considered = ['average_price', 'type', 'geography']

avocado_df = avocado_df[columns_considered]

avocado_df.head()
# print('Number of entries for various cities and regions:')

# print()



# for geographical_name in avocado_df.geography.unique():

#     num_entries = sum(avocado_df.geography == geographical_name)

#     print(f'{geographical_name:25} {num_entries}')
sub_df = avocado_df.query("type == 'conventional'")



plt.scatter(sub_df.index, sub_df.average_price, cmap='plasma')

plt.title('Average price of conventional avocados in all regions and ' \

          'cities over time')



plt.xlabel('Date')

plt.ylabel('Average price')

plt.show()
def plot_rolling_stats(time_series, window, avocado_type, geography):

    """

    A helper function for plotting the given time series, its rolling

    mean and standard deviation.

    """



    rolling_mean = time_series.rolling(window=window).mean()

    rolling_std = time_series.rolling(window=window).std()



    index = time_series.index



    sns.lineplot(x=index, y=time_series.average_price,

                 label='data', color='cornflowerblue')

    

    sns.lineplot(x=index, y=rolling_mean.average_price,

                 label='rolling mean', color='orange')

    

    sns.lineplot(x=index, y=rolling_std.average_price,

                 label='rolling std', color='seagreen')

    

    plt.title(f'Average price of {avocado_type} avocados in {geography}')

    plt.xlabel('Date')

    plt.ylabel('Average price')    
# NB: these two variables affect all the following calculations in that kernel.

AVOCADO_TYPE = 'conventional'

GEOGRAPHY = 'Total U.S.'



sub_df = avocado_df.query(f"type == '{AVOCADO_TYPE}' and " \

                          f"geography == '{GEOGRAPHY}'")

                          

sub_df.drop(['type', 'geography'], axis=1, inplace=True)

sub_df
# sub_df = sub_df.resample('2W').mean().bfill()

# sub_df.dropna(axis=0, inplace=True)

# sub_df
plot_rolling_stats(sub_df, window=4, avocado_type=AVOCADO_TYPE, 

                   geography=GEOGRAPHY)
# sub_df = sub_df.diff(periods=1)

# sub_df
# sub_df.dropna(axis=0, inplace=True)

# sub_df
# plot_rolling_stats(sub_df, window=4, avocado_type=AVOCADO_TYPE, region=REGION)
TEST_SET_SIZE = 45  # number of weeks left for the test set



data = sub_df.values

train_set, test_set = data[:-TEST_SET_SIZE], data[-TEST_SET_SIZE:]



print('shapes:', data.shape, train_set.shape, test_set.shape)
train_set_size = len(data) - TEST_SET_SIZE

train_set_dates = sub_df.head(train_set_size).index  # for plotting

test_set_dates = sub_df.tail(TEST_SET_SIZE).index  



plt.plot(train_set_dates, train_set, color='cornflowerblue', label='train data')

plt.plot(test_set_dates, test_set, color='orange', label='test data')

plt.legend(loc='best')

plt.title(f'Average price of {AVOCADO_TYPE} avocados in {GEOGRAPHY}')

plt.xlabel('Date')

plt.ylabel('Average price')

plt.show()
from statsmodels.tsa.ar_model import AutoReg



model = AutoReg(train_set, lags=52)  # use time span of 1 year for lagging

trained_model = model.fit()

# print('Coefficients: %s' % trained_model.params)
from sklearn.metrics import mean_squared_error as mse



predictions = trained_model.predict(start=train_set_size, 

                                    end=train_set_size + TEST_SET_SIZE - 1)



error = mse(test_set, predictions)



print(f'test MSE: {error:.3}')

print(f'test RMSE: {error ** 0.5:.3}')
plt.plot(test_set_dates, predictions, color='orange', label='predicted')

plt.plot(sub_df.index, sub_df.average_price, color='cornflowerblue', 

         label='ground truth')



plt.legend(loc='best')

plt.title(f'Average price of {AVOCADO_TYPE} avocados in {GEOGRAPHY}')

plt.xlabel('Date')

plt.ylabel('Average price')

plt.show()