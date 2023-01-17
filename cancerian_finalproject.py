# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import time



from tqdm import tqdm



import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
%%time

train = pd.read_csv("../input/train.csv", dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
print(train.shape)

print(train.head())
fig, ax = plt.subplots(2,1, figsize=(20,10))

ax[0].plot(train['acoustic_data'].values[::100], color='g')

ax[0].set_title("Acoustic data for 1% sample data")

ax[0].set_xlabel("Index")

ax[0].set_ylabel("Acoustic Data Signal");

ax[1].plot(train['time_to_failure'].values[::100], color='b')

ax[1].set_title("Time to Failure for 1% sample data")

ax[1].set_xlabel("Index")

ax[1].set_ylabel("Time to Failure in ms");
train.iloc[::50].describe()
def plotAroundPoints(start, end, ith):

    fig, ax1 = plt.subplots(figsize=(8, 4))

    plt.title("Trends of acoustic_data and time_to_failure around the {} earthquake".format(ith))

    plt.plot(train['acoustic_data'].values[start:end:50], color='b')

    ax1.set_ylabel('acoustic_data', color='b')

    plt.legend(['acoustic_data'])

    ax2 = ax1.twinx()

    plt.plot(train['time_to_failure'].values[start:end:50], color='g')

    ax2.set_ylabel('time_to_failure', color='g')

    plt.legend(['time_to_failure'], loc=(0.75, 0.1))



plotAroundPoints(0, 30000000, "first")

plotAroundPoints(30000000, 60000000, "second")

plotAroundPoints(90000000, 120000000, "third")

plotAroundPoints(125000000, 155000000, "fourth")

plotAroundPoints(170000000, 200000000, "fifth")

plotAroundPoints(200000000, 230000000, "sixth")

plotAroundPoints(225000000, 255000000, "seventh")

plotAroundPoints(285000000, 315000000, "eigth")

plotAroundPoints(325000000, 355000000, "ninth")

plotAroundPoints(360000000, 390000000, "tenth")

plotAroundPoints(405000000, 455000000, "eleventh")

plotAroundPoints(440000000, 470000000, "twelvth")

plotAroundPoints(480000000, 510000000, "thirteenth")

plotAroundPoints(510000000, 540000000, "fourteenth")

plotAroundPoints(560000000, 590000000, "fifteenth")

plotAroundPoints(605000000, 635000000, "sixteenth")

test_dir = "../input/test"

test_files = os.listdir(test_dir)

print(test_files[0:5])

print("Number of test files: {}".format(len(test_files)))

test_file_0 = pd.read_csv('../input/test/' + test_files[0])

print("Dimensions of the first test file: {}".format(test_file_0.shape))

test_file_0.head()
submission = pd.read_csv("../input/sample_submission.csv", index_col='seg_id', dtype={"time_to_failure": np.float32})

submission.head()
len(submission)
plt.scatter(train['acoustic_data'].values[::100], train['time_to_failure'].values[::100], s=10)

plt.show()
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import make_scorer

from sklearn.metrics import r2_score

from sklearn.metrics import median_absolute_error



X_train, X_test, y_train, y_test = train_test_split(train['acoustic_data'].values[::25].reshape(-1, 1), train['time_to_failure'].values[::25], test_size=0.2)



quake_linear_model = LinearRegression()

quake_linear_model.fit(X_train, y_train)



y_train_pred = quake_linear_model.predict(X_train)

y_test_pred = quake_linear_model.predict(X_test)



r2_train_score = r2_score(y_train, y_train_pred)

mae_train_score = median_absolute_error(y_train, y_train_pred)

r2_test_score = r2_score(y_test, y_test_pred)

mae_test_score = median_absolute_error(y_test, y_test_pred)



print("R2 score for training data: {} and for the test data: {}".format(r2_train_score, r2_test_score))

print("Mean Absolute Error score for training data: {} and for the test data: {}".format(mae_train_score, mae_test_score))
chunk_size = 150000



chunks = int(np.floor(train.shape[0]/chunk_size))



X_data = pd.DataFrame(index=range(chunks), dtype=np.float32, columns=['min','max','std', 'avg', 'sum', 'median', 'mean_diff', 

                                                                       'q05', 'q25', 'q75', 'q95'])

y_data = pd.DataFrame(index=range(chunks), dtype=np.float32, columns=['ttf'])



def create_features(data_chunk, X_df, chunk_no, col_name='acoustic_data'):

    x = data_chunk[col_name]

    X_df.loc[chunk_no, 'min'] = x.min()

    X_df.loc[chunk_no, 'max'] = x.max()

    X_df.loc[chunk_no, 'std'] = x.std()

    X_df.loc[chunk_no, 'avg'] = x.mean()

    X_df.loc[chunk_no, 'sum'] = x.sum()

    X_df.loc[chunk_no, 'median'] = x.median()

    X_df.loc[chunk_no, 'mean_diff'] = np.mean(np.diff(x))

    X_df.loc[chunk_no, 'q05'] = np.quantile(x, 0.05)

    X_df.loc[chunk_no, 'q25'] = np.quantile(x, 0.25)

    X_df.loc[chunk_no, 'q75'] = np.quantile(x, 0.75)

    X_df.loc[chunk_no, 'q95'] = np.quantile(x, 0.95)

    return X_df
for chunk_no in tqdm(range(chunks)):

    data_chunk = train.iloc[chunk_no*chunk_size:chunk_no*chunk_size+chunk_size]

    X_data = create_features(data_chunk, X_data, chunk_no)

    y = data_chunk['time_to_failure'].values[-1]

    y_data.loc[chunk_no, 'ttf'] = y
print(X_data.shape)

print(y_data.shape)

print(X_data.shape[1])

X_data.head()
X_train, X_test, y_train, y_test = train_test_split(X_data.values, y_data.values, test_size=0.2)

# X_test

# X_data.values

X_train.shape
from keras.models import Sequential

from keras.layers import Dense, Dropout, CuDNNGRU, CuDNNLSTM, Flatten

from keras.optimizers import adam

from keras.callbacks import ModelCheckpoint



model = Sequential()

# model.add(CuDNNLSTM(64, kernel_initializer="RandomUniform", input_shape= (X_train.shape[1], 1)))

model.add(CuDNNGRU(64, kernel_initializer="RandomUniform", input_shape= (X_train.shape[1], 1)))

model.add(Dropout(0.2))

model.add(Dense(64, activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(64, activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(64, activation="relu"))

model.add(Dropout(0.2))

# model.add(Flatten())

model.add(Dense(1))

model.summary()
from keras.callbacks import ModelCheckpoint



# Reshaping for fit

# X_train_array = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))

X_train_array = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

y_train_array = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))



# model.compile(loss="mean_squared_error", optimizer="rmsprop", metrics=["mse"])

model.compile(loss="mean_absolute_error", optimizer="adam", metrics=["mae", "mse"])



checkpointer = ModelCheckpoint("model.weights.hdf5", save_best_only=True, verbose=1)



build = model.fit(X_train_array, y_train, epochs=200, batch_size=30, validation_split = 0.20, callbacks=[checkpointer], verbose=1)
print(build.history.keys())
# summarize history for loss

plt.plot(build.history['loss'])

plt.plot(build.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper right')

plt.show()
X_train_array = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

predictions = model.predict(X_train_array)



r2_pred_score = r2_score(y_train, predictions)

mae_pred_score = median_absolute_error(y_train, predictions)



print("R2 score for training data: {}".format(r2_pred_score))

print("Mean Absolute Error score for training data: {}".format(mae_pred_score))
X_test_array = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predictions_test = model.predict(X_test_array)



r2_pred_test_score = r2_score(y_test, predictions_test)

mae_pred_test_score = median_absolute_error(y_test, predictions_test)



print("R2 score for test data: {}".format(r2_pred_test_score))

print("Mean Absolute Error score for test data: {}".format(mae_pred_test_score))
X_sub = pd.DataFrame(columns=X_data.columns, dtype=np.float32)



for i, seg_id in enumerate(tqdm(submission.index)):

    seg = pd.read_csv('../input/test/' + str(seg_id) + '.csv')

    X_seg = create_features(seg, X_sub, i)

    # print(X_seg)

    # X_seg_array = np.reshape(X_seg.values, (X_seg.shape[0], X_seg.shape[1], 1))

    # pred_seg = model.predict(X_seg_array)

    # print(pred_seg)

    # submission.time_to_failure[i] = pred_seg
print(X_sub.shape)

X_sub.head()
X_seg_array = np.reshape(X_seg.values, (X_seg.shape[0], X_seg.shape[1], 1))

pred_final = model.predict(X_seg_array)

submission['time_to_failure'] = pred_final

submission.head()
submission.to_csv('submission.csv')
possible_eq = submission.loc[submission["time_to_failure"] < 1.0]

print(possible_eq)

print(type(possible_eq))

print(possible_eq.columns)

# segments = ["seg_26a2a0", "seg_724df9", "seg_7a9f2b", "seg_7fa6ec", "seg_aa98cc", "seg_b35174", "seg_c80857", "seg_e3d751"]



for seg_id in possible_eq.index:

    print(seg_id)

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    fig, ax1 = plt.subplots(figsize=(8, 4))

    plt.title("Trends of acoustic_data for test file {}".format(seg_id))

    plt.plot(seg['acoustic_data'].values, color='b')

    ax1.set_ylabel('acoustic_data', color='b')

    plt.legend(['acoustic_data'])
