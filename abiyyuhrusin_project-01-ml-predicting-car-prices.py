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
data = pd.read_csv('/kaggle/input/imports-85.data')

data
cols = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 

        'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 

        'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-rate', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']



data = pd.read_csv('/kaggle/input/imports-85.data', names=cols)

pd.options.display.max_columns=99



data.head()
# selecting the columns with continuous numercial values, according to https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.names

cols1 = continuous_values_cols = ['normalized-losses', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'bore', 'stroke', 'compression-rate', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']

data = data[cols1]

data
# replace '?' values in 'normalized-losses' with NaN

data1 = data.copy() #prevent SettingWithCopyWarning

# data1 = data1.replace('?', np.nan)

data1 = data1.replace('?',np.nan)

data1
# convert all columns to float

data1 = data1.astype('float')

data1.info()
data1.isnull().sum()
# because the 'price' column is what we want to predict, then we'll remove any rows with missing value on 'price'

data1 = data1.dropna(subset=['price'])

data1.isnull().sum()
# replace missing values in other columns with the average value

data1 = data1.fillna(value=data1.mean())

data1.isnull().sum()
# normalize the dataset

data2 = (data1 - data1.min()) / (data1.max() - data1.min())

data2['price'] = data1['price']

data2
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error
def knn_train_test(training_col, target_col, df):

    # randomize order of rows

    np.random.seed(1)

    shuffled_index = np.random.permutation(df.index)

    df = df.reindex(shuffled_index)

    

    # split data set

    border = int(len(data1)/2)

    train = df.iloc[:border]

    test = df.iloc[border:]

    

    # fit knn model with default parameters

    knn = KNeighborsRegressor()

    knn.fit(train[[training_col]], train[target_col])

    

    # make prediction

    pred = knn.predict(test[[training_col]])

    

    # calculate mse and rmse

    mse = mean_squared_error(test[target_col], pred)

    rmse = mse**0.5

    return rmse



dct = {}

cols = ['normalized-losses', 'wheel-base', 'length', 'width', 'height',

       'curb-weight', 'bore', 'stroke', 'compression-rate', 'horsepower',

       'peak-rpm', 'city-mpg', 'highway-mpg']

for i in cols:

    result = knn_train_test(i, 'price', data2)

    dct[i] = result



# create series from the dictionary to allow sorting    

dct_srs = pd.Series(dct)

dct_srs.sort_values()
import matplotlib.pyplot as plt
# modify the function to to accept k values instead of a single default value

def knn_train_test(training_col, target_col, df):

    # randomize order of rows

    np.random.seed(1)

    shuffled_index = np.random.permutation(df.index)

    df = df.reindex(shuffled_index)

    

    # split data set

    border = int(len(data1)/2)

    train = df.iloc[:border]

    test = df.iloc[border:]

    

    # fit knn model with varying k values

    kvals = [1,3,5,7,9]

    rmses = {}

    for k in kvals:

        knn = KNeighborsRegressor(n_neighbors=k)

        knn.fit(train[[training_col]], train[target_col])

        

        # make prediction

        pred = knn.predict(test[[training_col]])

        

        # calculate mse and rmse

        mse = mean_squared_error(test[target_col], pred)

        rmse = mse**0.5

        rmses[k] = rmse

        

    return rmses



# iterate the modified function on every column except price

cols = ['normalized-losses', 'wheel-base', 'length', 'width', 'height',

       'curb-weight', 'bore', 'stroke', 'compression-rate', 'horsepower',

       'peak-rpm', 'city-mpg', 'highway-mpg']



dct = {}

for i in cols:

    result = knn_train_test(i, 'price', data2)

    dct[i] = result



dct
# plot the data

for i in dct.keys():

    plt.plot(list(dct[i].keys()), list(dct[i].values()))

    plt.xlabel('k value')

    plt.ylabel('rmse')
# calculate the average RMSE across different k values for each feature

dct2 = {}

for k,v in dct.items():

    values = list(v.values())

    dct2[k] = np.mean(values)



# create a series from the dictionary to allow sorting

dct2_ser = pd.Series(dct2)

dct_sort = dct2_ser.sort_values()

dct_sort
# modify the function to accept list of column names (instead of only string)

def knn_train_test(training_col, target_col, df):

    # randomize order of rows

    np.random.seed(1)

    shuffled_index = np.random.permutation(df.index)

    df = df.reindex(shuffled_index)

    

    # split data set

    border = int(len(data1)/2)

    train = df.iloc[:border]

    test = df.iloc[border:]

    

    # fit knn model with default parameters

    knn = KNeighborsRegressor()

    knn.fit(train[training_col], train[target_col])

    

    # make prediction

    pred = knn.predict(test[training_col])

    

    # calculate mse and rmse

    mse = mean_squared_error(test[target_col], pred)

    rmse = mse**0.5

    return rmse



# calculate the rmse using different number of features

best_ft = {}

for i in range(2,6):

    best_ft[f'{i} best features'] = knn_train_test(dct_sort.index[:i], 'price', data2)

    

best_ft
def knn_train_test(training_col, target_col, df):

    # randomize order of rows

    np.random.seed(1)

    shuffled_index = np.random.permutation(df.index)

    df = df.reindex(shuffled_index)

    

    # split data set

    border = int(len(data1)/2)

    train = df.iloc[:border]

    test = df.iloc[border:]

    

    # fit knn model with varying k values

    kvals = list(range(1,26,2))

    rmses = {}

    for k in kvals:

        knn = KNeighborsRegressor(n_neighbors=k)

        knn.fit(train[training_col], train[target_col])

        pred = knn.predict(test[training_col])

        mse = mean_squared_error(test[target_col], pred)

        rmse = mse**0.5

        rmses[k] = rmse

    return rmses

        

# calculate the rmse using the best three models (2,3, and 4 features)

best_ft = {}

for i in range(2,5):

    best_ft[f'{i} best features'] = knn_train_test(dct_sort.index[:i], 'price', data2)

    

best_ft
# plot the data

for k,v in best_ft.items():

    x = list(v.keys())

    y = list(v.values())

    plt.plot(x,y, label=k)

    plt.xticks(np.arange(min(x), max(x)+1, 2.0))

    plt.legend()
# modify the model to accept k value of 1

def knn_train_test(training_col, target_col, df):

    # randomize order of rows

    np.random.seed(1)

    shuffled_index = np.random.permutation(df.index)

    df = df.reindex(shuffled_index)

    

    # split data set

    border = int(len(data1)/2)

    train = df.iloc[:border]

    test = df.iloc[border:]

    

    # fit knn model with k value=1

    knn = KNeighborsRegressor(n_neighbors=1)

    knn.fit(train[training_col], train[target_col])

    pred = knn.predict(test[training_col])

    return pred



# run the model with 4 best features

cols = list(dct_sort.keys()[:4])

predicted_price = knn_train_test(cols, 'price', data2)



# compare the prediction with test data

border = int(len(data2)/2)

test = data2.iloc[border:]

test['predicted'] = predicted_price

test