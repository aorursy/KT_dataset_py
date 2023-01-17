# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#import the required libraries

import pandas as pd



import numpy as np



import matplotlib.pyplot as plt



import seaborn as sns



from tqdm import tqdm

from tqdm import tqdm_notebook



from scipy.signal import hilbert

from scipy.signal import hann

from scipy.signal import convolve
train_data = pd.read_csv('../input/train.csv', dtype = {'acoustic_data': np.int16, 'time_to_failure': np.float32})
#let's look at the training data

print(train_data.head())

print(100*'-')

print(train_data.shape)
#let's prepare our data

seg_length = 150000

total_samples = int(np.floor((train_data.shape[0]) / seg_length))



x_train = pd.DataFrame(index = range(total_samples), dtype = np.float64) #an empty dataframe holding our feature values

y_train = pd.DataFrame(index = range(total_samples), columns = ['time_to_failure'], dtype = np.float64) #an empty dataframe holding our target labels
def calc_change_rate(x):

    change = (np.diff(x) / x[:-1]).values

    change = change[np.nonzero(change)[0]]

    change = change[~np.isnan(change)]

    change = change[change != -np.inf]

    change = change[change != np.inf]

    return np.mean(change)



for value in tqdm_notebook(range(total_samples)):

    sample = train_data.iloc[value*seg_length : value*seg_length + seg_length]

    x = pd.Series(sample['acoustic_data'].values)

    y = sample['time_to_failure'].values[-1]

    y_train.loc[value, 'time_to_failure'] = y

    

    x_train.loc[value, 'AVERAGE'] = x.mean()

    x_train.loc[value, 'STD'] = x.std()

    x_train.loc[value, 'MAX'] = x.max()

    x_train.loc[value, 'MIN'] = x.min() 

    x_train.loc[value, 'SUM'] = x.sum()

    

    x_train.loc[value, 'MEAN_CHANGE_ABS'] = np.mean(np.diff(x))

    x_train.loc[value, 'MEAN_CHANGE_RATE'] = calc_change_rate(x)

    

    x_train.loc[value, 'MAX_TO_MIN'] = x.max() / np.abs(x.min())

    x_train.loc[value, 'MAX_TO_MIN_DIFF'] = x.max() - np.abs(x.min())

    x_train.loc[value, 'COUNT_BIG'] = len(x[np.abs(x) > 500])

    

    x_train.loc[value, 'AVERAGE_FIRST_10000'] = x[:10000].mean()

    x_train.loc[value, 'AVERAGE_LAST_10000']  =  x[-10000:].mean()

    x_train.loc[value, 'AVERAGE_FIRST_50000'] = x[:50000].mean()

    x_train.loc[value, 'AVERAGE_LAST_50000'] = x[-50000:].mean()

    

    x_train.loc[value, 'STD_FIRST_10000'] = x[:10000].std()

    x_train.loc[value, 'STD_LAST_10000']  =  x[-10000:].std()

    x_train.loc[value, 'STD_FIRST_50000'] = x[:50000].std()

    x_train.loc[value, 'STD_LAST_50000'] = x[-50000:].std()

    

    x_train.loc[value, 'ABS_AVERAGE'] = np.abs(x).mean()

    x_train.loc[value, 'ABS_STD'] = np.abs(x).std()

    x_train.loc[value, 'ABS_MAX'] = np.abs(x).max()

    x_train.loc[value, 'ABS_MIN'] = np.abs(x).min()

    

    x_train.loc[value, '10Q'] = np.percentile(x, 0.10)

    x_train.loc[value, '25Q'] = np.percentile(x, 0.25)

    x_train.loc[value, '50Q'] = np.percentile(x, 0.50)

    x_train.loc[value, '75Q'] = np.percentile(x, 0.75)

    x_train.loc[value, '90Q'] = np.percentile(x, 0.90)

    

    x_train.loc[value, 'ABS_1Q'] = np.percentile(x, np.abs(0.01))

    x_train.loc[value, 'ABS_5Q'] = np.percentile(x, np.abs(0.05))

    x_train.loc[value, 'ABS_30Q'] = np.percentile(x, np.abs(0.30))

    x_train.loc[value, 'ABS_60Q'] = np.percentile(x, np.abs(0.60))

    x_train.loc[value, 'ABS_95Q'] = np.percentile(x, np.abs(0.95))

    x_train.loc[value, 'ABS_99Q'] = np.percentile(x, np.abs(0.99))

    

    x_train.loc[value, 'KURTOSIS'] = x.kurtosis()

    x_train.loc[value, 'SKEW'] = x.skew()

    x_train.loc[value, 'MEDIAN'] = x.median()

    

    x_train.loc[value, 'HILBERT_MEAN'] = np.abs(hilbert(x)).mean()

    x_train.loc[value, 'HANN_WINDOW_MEAN'] = (convolve(x, hann(150), mode = 'same') / sum(hann(150))).mean()

    

    for windows in [10, 100, 1000]:

        x_roll_std = x.rolling(windows).std().dropna().values

        x_roll_mean = x.rolling(windows).mean().dropna().values

        

        x_train.loc[value, 'AVG_ROLL_STD' + str(windows)] = x_roll_std.mean()

        x_train.loc[value, 'STD_ROLL_STD' + str(windows)] = x_roll_std.std()

        x_train.loc[value, 'MAX_ROLL_STD' + str(windows)] = x_roll_std.max()

        x_train.loc[value, 'MIN_ROLL_STD' + str(windows)] = x_roll_std.min()

        x_train.loc[value, '1Q_ROLL_STD' + str(windows)] = np.quantile(x_roll_std, 0.01)

        x_train.loc[value, '5Q_ROLL_STD' + str(windows)] = np.quantile(x_roll_std, 0.05)

        x_train.loc[value, '95Q_ROLL_STD' + str(windows)] = np.quantile(x_roll_std, 0.95)

        x_train.loc[value, '99Q_ROLL_STD' + str(windows)] = np.quantile(x_roll_std, 0.99)

        x_train.loc[value, 'AV_CHANGE_ABS_ROLL_STD' + str(windows)] = np.mean(np.diff(x_roll_std))

        x_train.loc[value, 'ABS_MAX_ROLL_STD' + str(windows)] = np.abs(x_roll_std).max()

        

        x_train.loc[value, 'AVG_ROLL_MEAN' + str(windows)] = x_roll_mean.mean()

        x_train.loc[value, 'STD_ROLL_MEAN' + str(windows)] = x_roll_mean.std()

        x_train.loc[value, 'MAX_ROLL_MEAN' + str(windows)] = x_roll_mean.max()

        x_train.loc[value, 'MIN_ROLL_MEAN' + str(windows)] = x_roll_mean.min()

        x_train.loc[value, '1Q_ROLL_MEAN' + str(windows)] = np.quantile(x_roll_mean, 0.01)

        x_train.loc[value, '5Q_ROLL_MEAN' + str(windows)] = np.quantile(x_roll_mean, 0.05)

        x_train.loc[value, '95Q_ROLL_MEAN' + str(windows)] = np.quantile(x_roll_mean, 0.95)

        x_train.loc[value, '99Q_ROLL_MEAN' + str(windows)] = np.quantile(x_roll_mean, 0.99)

        x_train.loc[value, 'AV_CHANGE_ABS_ROLL_MEAN' + str(windows)] = np.mean(np.diff(x_roll_mean))

        x_train.loc[value, 'ABS_MAX_ROLL_MEAN' + str(windows)] = np.abs(x_roll_mean).max()
x_train.head()
y_train.head(10) #our training dataframe holding our output labels
y_train.isnull().sum()
print(x_train.shape)

print(y_train.shape)
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_absolute_error
#normalizing the data

scaler = StandardScaler()

scaler.fit(x_train)

x_train_scaled = scaler.transform(x_train)



y_train_flatten = y_train.values.ravel() #flattening the y_train
#let's look at the mormalized data

x_train_dataframe = pd.DataFrame(x_train_scaled)

x_train_dataframe.head(10)
#splitting the data into train and test

x_train1, x_test1, y_train1, y_test1 = train_test_split(x_train, y_train, test_size = 0.2, random_state = 101)
#normalizing the data

scaler = StandardScaler()

scaler.fit(x_train1)

x_train_scaled1 = scaler.transform(x_train1)

x_test_scaled1 = scaler.transform(x_test1)



y_train_flatten1 = y_train1.values.ravel() #flattening the y_train

y_test_flatten1 = y_test1.values.ravel() #flattening the y_test
print(x_train_scaled1.shape)

print(x_test_scaled1.shape)

print(y_train_flatten1.shape)

print(y_test_flatten1.shape)
#import the libraries for building the neural net

import keras

from keras.layers import Dense

from keras.layers import BatchNormalization

from keras.layers import Activation

from keras.layers import Dropout

from keras.models import Sequential

from keras import optimizers
kernel_init = 'normal'

def Model(input_dim, activation, classes):

    model = Sequential()



    model.add(Dense(1024, kernel_initializer = kernel_init, input_dim = input_dim))

    model.add(Activation(activation))

    model.add(BatchNormalization())

#     model.add(Dense(1024, kernel_initializer = kernel_init, input_dim = input_dim))

#     model.add(Activation(activation))

#     model.add(BatchNormalization())

    model.add(Dropout(0.3))



    model.add(Dense(512, kernel_initializer = kernel_init)) 

    model.add(Activation(activation))

    model.add(BatchNormalization())

#     model.add(Dense(512, kernel_initializer = kernel_init)) 

#     model.add(Activation(activation))

#     model.add(BatchNormalization())

    model.add(Dropout(0.3))



    model.add(Dense(64, kernel_initializer = kernel_init))    

    model.add(Activation(activation))

    model.add(BatchNormalization())

#     model.add(Dense(64, kernel_initializer = kernel_init))    

#     model.add(Activation(activation))

#     model.add(BatchNormalization())

    model.add(Dropout(0.3))

    

#     model.add(Dense(32, kernel_initializer = kernel_init))    

#     model.add(Activation(activation))

#     model.add(BatchNormalization())

#     model.add(Dense(32, kernel_initializer = kernel_init))    

#     model.add(Activation(activation))

#     model.add(BatchNormalization())

#     model.add(Dropout(0.2))



    model.add(Dense(classes, kernel_initializer = kernel_init))    

    model.add(Activation('linear'))

    

    return model
input_dim = x_train.shape[1]

activation = 'tanh'

classes = 1 #the output labels

model = Model(input_dim = input_dim, activation = activation, classes = classes)



#model.summary()
#compile the model

optim = optimizers.Adam(lr = 0.001, decay = 0.001 / 32)

model.compile(loss = 'mean_absolute_error', optimizer = optim, metrics = ['mean_absolute_error'])
#fit the model onto the dataset

h = model.fit(x_train_scaled1, y_train_flatten1, epochs = 100, batch_size = 32, verbose = 1)
#making predictions

prediction_score = []

mae_scores = []

predictions = model.predict(x_test_scaled1, verbose = 1, batch_size = 32) #make predictons on the test data 

prediction_score.append(predictions)



mae = mean_absolute_error(predictions, y_test_flatten1) #calculating MAE between our actual output labels and the predicted otuput labels

mae_scores.append(mae)
print(np.mean(mae_scores)) #average of mean absolute errors
predictions = np.hstack([p.reshape(-1,1) for p in prediction_score])

predictions = np.mean(predictions, axis = 1)

print(mean_absolute_error(predictions, y_test1))
def calc_change_rate(x):

    change = (np.diff(x) / x[:-1]).values

    change = change[np.nonzero(change)[0]]

    change = change[~np.isnan(change)]

    change = change[change != -np.inf]

    change = change[change != np.inf]

    return np.mean(change)



submission = pd.read_csv('../input/sample_submission.csv', index_col = 'seg_id')

X_test = pd.DataFrame(columns = x_train.columns, dtype = np.float64, index = submission.index)



for id in tqdm_notebook(X_test.index):

    seg = pd.read_csv('../input/test/' + id + '.csv')

    

    x = pd.Series(seg['acoustic_data'].values)

    

    X_test.loc[id, 'AVERAGE'] = x.mean()

    X_test.loc[id, 'STD'] = x.std()

    X_test.loc[id, 'MAX'] = x.max()

    X_test.loc[id, 'MIN'] = x.min() 

    X_test.loc[id, 'SUM'] = x.sum()

    

    X_test.loc[id, 'MEAN_CHANGE_ABS'] = np.mean(np.diff(x))

    X_test.loc[id, 'MEAN_CHANGE_RATE'] = calc_change_rate(x)

    

    X_test.loc[id, 'MAX_TO_MIN'] = x.max() / np.abs(x.min())

    X_test.loc[id, 'MAX_TO_MIN_DIFF'] = x.max() - np.abs(x.min())

    X_test.loc[id, 'COUNT_BIG'] = len(x[np.abs(x) > 500])

    

    X_test.loc[id, 'AVERAGE_FIRST_10000'] = x[:10000].mean()

    X_test.loc[id, 'AVERAGE_LAST_10000']  =  x[-10000:].mean()

    X_test.loc[id, 'AVERAGE_FIRST_50000'] = x[:50000].mean()

    X_test.loc[id, 'AVERAGE_LAST_50000'] = x[-50000:].mean()

    

    X_test.loc[id, 'STD_FIRST_10000'] = x[:10000].std()

    X_test.loc[id, 'STD_LAST_10000']  =  x[-10000:].std()

    X_test.loc[id, 'STD_FIRST_50000'] = x[:50000].std()

    X_test.loc[id, 'STD_LAST_50000'] = x[-50000:].std()

    

    X_test.loc[id, 'ABS_AVERAGE'] = np.abs(x).mean()

    X_test.loc[id, 'ABS_STD'] = np.abs(x).std()

    X_test.loc[id, 'ABS_MAX'] = np.abs(x).max()

    X_test.loc[id, 'ABS_MIN'] = np.abs(x).min()

    

    X_test.loc[id, '10Q'] = np.percentile(x, 0.10)

    X_test.loc[id, '25Q'] = np.percentile(x, 0.25)

    X_test.loc[id, '50Q'] = np.percentile(x, 0.50)

    X_test.loc[id, '75Q'] = np.percentile(x, 0.75)

    X_test.loc[id, '90Q'] = np.percentile(x, 0.90)

    

    X_test.loc[id, 'ABS_1Q'] = np.percentile(x, np.abs(0.01))

    X_test.loc[id, 'ABS_5Q'] = np.percentile(x, np.abs(0.05))

    X_test.loc[id, 'ABS_30Q'] = np.percentile(x, np.abs(0.30))

    X_test.loc[id, 'ABS_60Q'] = np.percentile(x, np.abs(0.60))

    X_test.loc[id, 'ABS_95Q'] = np.percentile(x, np.abs(0.95))

    X_test.loc[id, 'ABS_99Q'] = np.percentile(x, np.abs(0.99))

    

    X_test.loc[id, 'KURTOSIS'] = x.kurtosis()

    X_test.loc[id, 'SKEW'] = x.skew()

    X_test.loc[id, 'MEDIAN'] = x.median()

    

    X_test.loc[id, 'HILBERT_MEAN'] = np.abs(hilbert(x)).mean()

    X_test.loc[id, 'HANN_WINDOW_MEAN'] = (convolve(x, hann(150), mode = 'same') / sum(hann(150))).mean()

    

    for windows in [10, 100, 1000]:

        x_roll_std = x.rolling(windows).std().dropna().values

        x_roll_mean = x.rolling(windows).mean().dropna().values

        

        X_test.loc[id, 'AVG_ROLL_STD' + str(windows)] = x_roll_std.mean()

        X_test.loc[id, 'STD_ROLL_STD' + str(windows)] = x_roll_std.std()

        X_test.loc[id, 'MAX_ROLL_STD' + str(windows)] = x_roll_std.max()

        X_test.loc[id, 'MIN_ROLL_STD' + str(windows)] = x_roll_std.min()

        X_test.loc[id, '1Q_ROLL_STD' + str(windows)] = np.quantile(x_roll_std, 0.01)

        X_test.loc[id, '5Q_ROLL_STD' + str(windows)] = np.quantile(x_roll_std, 0.05)

        X_test.loc[id, '95Q_ROLL_STD' + str(windows)] = np.quantile(x_roll_std, 0.95)

        X_test.loc[id, '99Q_ROLL_STD' + str(windows)] = np.quantile(x_roll_std, 0.99)

        X_test.loc[id, 'AV_CHANGE_ABS_ROLL_STD' + str(windows)] = np.mean(np.diff(x_roll_std))

        X_test.loc[id, 'ABS_MAX_ROLL_STD' + str(windows)] = np.abs(x_roll_std).max()

        

        X_test.loc[id, 'AVG_ROLL_MEAN' + str(windows)] = x_roll_mean.mean()

        X_test.loc[id, 'STD_ROLL_MEAN' + str(windows)] = x_roll_mean.std()

        X_test.loc[id, 'MAX_ROLL_MEAN' + str(windows)] = x_roll_mean.max()

        X_test.loc[id, 'MIN_ROLL_MEAN' + str(windows)] = x_roll_mean.min()

        X_test.loc[id, '1Q_ROLL_MEAN' + str(windows)] = np.quantile(x_roll_mean, 0.01)

        X_test.loc[id, '5Q_ROLL_MEAN' + str(windows)] = np.quantile(x_roll_mean, 0.05)

        X_test.loc[id, '95Q_ROLL_MEAN' + str(windows)] = np.quantile(x_roll_mean, 0.95)

        X_test.loc[id, '99Q_ROLL_MEAN' + str(windows)] = np.quantile(x_roll_mean, 0.99)

        X_test.loc[id, 'AV_CHANGE_ABS_ROLL_MEAN' + str(windows)] = np.mean(np.diff(x_roll_mean))

        X_test.loc[id, 'ABS_MAX_ROLL_MEAN' + str(windows)] = np.abs(x_roll_mean).max()
X_test.head(10)
print(X_test.shape)
X_test.isnull().sum()
submission.shape
#normalizing our testing data

X_test_scaled = scaler.transform(X_test)

print(X_test_scaled.shape)
input_dim = x_train.shape[1]

activation = 'tanh'

classes = 1



history = dict() #dictionery to store the history of individual models for later visualization

prediction_scores = dict() #dictionery to store the predicted scores of individual models on the test dataset



#here we will be training the same model for a total of 10 times and will be considering the mean of the output values for predictions

for i in np.arange(0, 15):

    optim = optimizers.Adam(lr = 0.001, decay = 0.001 / 32)

    ensemble_model = Model(input_dim = input_dim, activation = activation, classes = classes)

    ensemble_model.compile(loss = 'mean_absolute_error', optimizer = optim, metrics = ['mean_absolute_error'])

    print('TRAINING MODEL NO : {}'.format(i))

    H = ensemble_model.fit(x_train_scaled, y_train_flatten,

                  batch_size = 32,

                  epochs = 200,

                  verbose = 1)

    history[i] = H

    

#     ensemble_model.save('MODEL_{}.model'.format(i))

    

    predictions = ensemble_model.predict(X_test_scaled, verbose = 1, batch_size = 32)

    prediction_scores[i] = predictions
#making predictions

prediction1 = np.hstack([p.reshape(-1,1) for p in prediction_scores.values()]) #taking the scores of all the trained models

prediction1 = np.mean(prediction1, axis = 1)



print(prediction1.shape)
#submitting the file

submission['time_to_failure'] = prediction1

submission.to_csv('submission.csv')