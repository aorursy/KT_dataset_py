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
data = pd.read_csv("/kaggle/input/aapl-data/AAPL_data.csv")

data.info()


# using https://towardsdatascience.com/predicting-stock-price-with-lstm-13af86a74944

from matplotlib import pyplot as plt

plt.figure()

plt.plot(data["open"])

plt.plot(data["high"])

plt.plot(data["low"])

plt.plot(data["close"])

plt.title('S&P500 stock price history')

plt.ylabel('Price (USD)')

plt.xlabel('Days')

plt.legend(['open','high','low','close'], loc='upper left')

fig = plt.gcf()



fig.set_size_inches(27, 10.5)

plt.show()
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split



train_cols = ["open","high","low","close"]

df_train, df_test = train_test_split(data, train_size=0.8, test_size=0.2, shuffle=False)

print("Train and Test size", len(df_train), len(df_test))

# scale the feature MinMax, build array



# loc searches dataframe by label name... we are providing all the labels in train_cols and with : we mean we want every row to be returned

#values gets rid of the labels and indexes and returns a matrix instead of the dataframe

x = df_train.loc[:,train_cols].values

min_max_scaler = MinMaxScaler()

x_train = min_max_scaler.fit_transform(x)

x_test = min_max_scaler.transform(df_test.loc[:,train_cols])
BATCH_SIZE = 2

#3 days in the past 

TIME_STEPS = 3
def build_timeseries(mat, y_col_index):

    # y_col_index is the index of column that would act as output column

    # total number of time-series samples would be len(mat) - TIME_STEPS

    print("mat shape[0]: ", mat.shape[0])

    print("mat shape[1]: ", mat.shape[1])

    dim_0 = mat.shape[0] - TIME_STEPS

    dim_1 = mat.shape[1]

    x = np.zeros((dim_0, TIME_STEPS, dim_1))

    print("x.shape (-TIME_STEPS applied to second dimension): ", x.shape)

    y = np.zeros((dim_0,))

    

    for i in tqdm_notebook(range(dim_0)):

        x[i] = mat[i:TIME_STEPS+i]

        y[i] = mat[TIME_STEPS+i, y_col_index]

    print("length of time-series i/o",x.shape,y.shape)

    return x, y
build_timeseries(x_train, 3)