# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.model_selection import ShuffleSplit

from sklearn.linear_model import HuberRegressor, LinearRegression

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



from tqdm import tnrange,tqdm_notebook

from sklearn.preprocessing import StandardScaler

from sklearn.svm import NuSVR, SVR

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import ShuffleSplit, KFold

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import ElasticNet

from sklearn.svm import SVR



from scipy.signal import hilbert

from scipy.signal import hann

from scipy.signal import convolve

from scipy import stats



import matplotlib.pyplot as plt

import gc



import xgboost as xgb

import seaborn as sns

from xgboost import XGBRegressor

# Any results you write to the current directory are saved as output.
print("train shape", train.shape)
pd.set_option("display.precision", 15)  # show more decimals
def featurize(x):

    X = pd.Series()

    

    X['mean'] = x.mean()

    X['std'] = x.std()

    X['max'] = x.max()

    X['min'] = x.min()





    X['mean_change_abs'] = np.mean(np.diff(x))

    X['mean_change_rate'] = np.mean(np.nonzero((np.diff(x) / x[:-1]))[0])



    X['max_to_min'] = x.max() / np.abs(x.min())

    X['max_to_min_diff'] = x.max() - np.abs(x.min())

    X['count_big'] = len(x[np.abs(x) > 500])

    X['sum'] = x.sum()



    X['q95'] = np.quantile(x, 0.95)

    X['q99'] = np.quantile(x, 0.99)

    X['q05'] = np.quantile(x, 0.05)

    X['q01'] = np.quantile(x, 0.01)



    X['mad'] = x.mad()

    X['kurt'] = x.kurtosis()

    X['skew'] = x.skew()

    X['med'] = x.median()



    X['iqr'] = np.subtract(*np.percentile(x, [75, 25]))

    X['q999'] = np.quantile(x,0.999)

    X['q001'] = np.quantile(x,0.001)

    X['ave10'] = stats.trim_mean(x, 0.1)



    '''for windows in [10, 100, 1000]:

        if len(x) < windows:

            continue

        x_roll_std = x.rolling(windows).std().dropna().values

        x_roll_mean = x.rolling(windows).mean().dropna().values



        X['ave_roll_std_' + str(windows)] = x_roll_std.mean()

        X['std_roll_std_' + str(windows)] = x_roll_std.std()

        X['max_roll_std_' + str(windows)] = x_roll_std.max()

        X['min_roll_std_' + str(windows)] = x_roll_std.min()

        X['q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)

        X['q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)

        X['q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)

        X['q99_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.99)

        X['av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))

        X['av_change_rate_roll_std_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])

        X['abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()



        X['ave_roll_mean_' + str(windows)] = x_roll_mean.mean()

        X['std_roll_mean_' + str(windows)] = x_roll_mean.std()

        X['max_roll_mean_' + str(windows)] = x_roll_mean.max()

        X['min_roll_mean_' + str(windows)] = x_roll_mean.min()

        X['q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)

        X['q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)

        X['q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)

        X['q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.99)

        X['av_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))

        X['av_change_rate_roll_mean_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])

        X['abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()'''

    #print(X.shape)

    return X.values
def featurize_ts(x, n_steps=150):

    res = []

    for i in range(x.shape[0]):

        series = pd.Series(x[i, :])

        #print(series.shape)

        res.append(featurize(series))

    arr = np.array(res)

    #print(arr.shape)

    return arr
def create_X(x, last_index=None, n_steps=150, step_length=1000):

    if last_index == None:

        last_index=len(x)

       

    assert last_index - n_steps * step_length >= 0



    # Reshaping and approximate standardization with mean 5 and std 3.

    # ORIGINAL: I changed this becuase I got an No OpKernel was registered to support Op 'CudnnRNN' error

    #temp = (x[(last_index - n_steps * step_length):last_index].reshape(n_steps, -1) - 5 ) / 3

    # MY CHANGE: This doesn't fix things, I get the same errors

    temp = (x[(last_index - n_steps * step_length):last_index].values.reshape(n_steps, -1))#.astype(np.float32) - 5 ) / 3

    #print(temp.shape)

    

    # Extracts features of sequences of full length 1000, of the last 100 values and finally also 

    # of the last 10 observations. 

    return np.c_[featurize_ts(temp),

                 featurize_ts(temp[:, -step_length // 10:]),

                 featurize_ts(temp[:, -step_length // 100:])]
float_data = train
#https://gist.github.com/platdrag/e755f3947552804c42633a99ffd325d4

import threading

'''

    A generic iterator and generator that takes any iterator and wrap it to make it thread safe.

    This method was introducted by Anand Chitipothu in http://anandology.com/blog/using-iterators-and-generators/

    but was not compatible with python 3. This modified version is now compatible and works both in python 2.8 and 3.0 

'''

class threadsafe_iter:

    """Takes an iterator/generator and makes it thread-safe by

    serializing call to the `next` method of given iterator/generator.

    """

    def __init__(self, it):

        self.it = it

        self.lock = threading.Lock()



    def __iter__(self):

        return self



    def __next__(self):

        with self.lock:

            return self.it.__next__()



def threadsafe_generator(f):

    """A decorator that takes a generator function and makes it thread-safe.

    """

    def g(*a, **kw):

        return threadsafe_iter(f(*a, **kw))

    return g
# Query "create_X" to figure out the number of features

n_features = create_X(float_data.loc[0:150000, :]).shape[1]

print("Our RNN is based on %i features"% n_features)



# The generator endlessly selects "batch_size" ending positions of sub-time series. For each ending position,

# the "time_to_failure" serves as target, while the features are created by the function "create_X".

@threadsafe_generator

def generator(data, min_index=0, max_index=None, batch_size=16, n_steps=150, step_length=1000):

    if max_index is None:

        max_index = len(data) - 1

     

    while True:

        # Pick indices of ending positions

        rows = np.random.randint(min_index + n_steps * step_length, max_index, size=batch_size)

         

        # Initialize feature matrices and targets

        samples = np.zeros((batch_size, n_steps, n_features))

        targets = np.zeros(batch_size, )

        

        for j, row in enumerate(rows):

            samples[j] = create_X(data.iloc[:, 0], last_index=row, n_steps=n_steps, step_length=step_length)

            targets[j] = data.iloc[row - 1, 1]

        yield samples, targets

        

batch_size = 32



# Position of second (of 16) earthquake. Used to have a clean split

# between train and validation

second_earthquake = 50085877

float_data.iloc[second_earthquake, 1]



# Initialize generators

# train_gen = generator(float_data, batch_size=batch_size) # Use this for better score

train_gen = generator(float_data, batch_size=batch_size, min_index=second_earthquake + 1)

valid_gen = generator(float_data, batch_size=batch_size, max_index=second_earthquake)
from keras.models import Sequential

from keras.layers import Dense, GRU

from keras.optimizers import adam

from keras.callbacks import ModelCheckpoint



cb = [ModelCheckpoint("model.hdf5", save_best_only=True, period=3)]



model = Sequential()

model.add(GRU(48, input_shape=(None, n_features)))

model.add(Dense(10, activation='relu'))

model.add(Dense(1))



model.summary()
model.compile(optimizer=adam(lr=0.0005), loss="mae")



history = model.fit_generator(train_gen,

                              steps_per_epoch=1000,

                              epochs=30,

                              verbose=0,

                              callbacks=cb,

                              validation_data=valid_gen,

                              validation_steps=200)
import matplotlib.pyplot as plt



def perf_plot(history, what = 'loss'):

    x = history.history[what]

    val_x = history.history['val_' + what]

    epochs = np.asarray(history.epoch) + 1

    

    plt.plot(epochs, x, 'bo', label = "Training " + what)

    plt.plot(epochs, val_x, 'b', label = "Validation " + what)

    plt.title("Training and validation " + what)

    plt.xlabel("Epochs")

    plt.legend()

    plt.show()

    return None



perf_plot(history)