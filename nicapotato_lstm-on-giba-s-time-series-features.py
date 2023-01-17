import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

import lightgbm as lgb
from sklearn.model_selection import *
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.stats import mode, skew, kurtosis, entropy
from sklearn.ensemble import ExtraTreesRegressor

# Keras Neural Net / LSTM (RNN)
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, ThresholdedReLU, MaxPooling2D, Embedding, Dropout
from keras.optimizers import Adam, SGD, RMSprop
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns

import dask.dataframe as dd
from dask.multiprocessing import get

from tqdm import tqdm, tqdm_notebook
tqdm.pandas(tqdm_notebook)
from IPython.display import display

Debug = False
nrows = None
if Debug is True: nrows = 200
train = pd.read_csv("../input/train.csv", nrows=nrows)
test = pd.read_csv("../input/test.csv", nrows=nrows)
traindex = train.ID.copy()
testdex = test.ID.copy()
print(test.shape)

transact_cols = [f for f in train.columns if f not in ["ID", "target"]]
y = np.log1p(train["target"]).values
cols = ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1', '15ace8c9f', 
        'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9', 'd6bb78916', 'b43a7cfd5',
        '58232a6fb', '1702b5bf0', '324921c7b', '62e59a501', '2ec5b290f', '241f0f867',
        'fb49e4212', '66ace2992', 'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7',
        '1931ccfdd', '703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a',
        '6619d81fc', '1db387535']

from multiprocessing import Pool
CPU_CORES = 1
def _get_leak(df, cols, lag=0):
    """ To get leak value, we do following:
       1. Get string of all values after removing first two time steps
       2. For all rows we shift the row by two steps and again make a string
       3. Just find rows where string from 2 matches string from 1
       4. Get 1st time step of row in 3 (Currently, there is additional condition to only fetch value if we got exactly one match in step 3)"""
    series_str = df[cols[lag+2:]].apply(lambda x: "_".join(x.round(2).astype(str)), axis=1)
    series_shifted_str = df[cols].shift(lag+2, axis=1)[cols[lag+2:]].apply(lambda x: "_".join(x.round(2).astype(str)), axis=1)
    target_rows = series_shifted_str.progress_apply(lambda x: np.where(x == series_str)[0])
    target_vals = target_rows.apply(lambda x: df.loc[x[0], cols[lag]] if len(x)==1 else 0)
    return target_vals

def get_all_leak(df, cols=None, nlags=15):
    """
    We just recursively fetch target value for different lags
    """
    df =  df.copy()
    for i in range(nlags):
        print("Processing lag {}".format(i))
        df["leaked_target_"+str(i)] = _get_leak(df, cols, i)
    return df

test["target"] = train["target"].mean()
all_df = pd.concat([train[["ID", "target"] + cols], test[["ID", "target"]+ cols]]).reset_index(drop=True)
NLAGS = 15 #Increasing this might help push score a bit
all_df = get_all_leak(all_df, cols=cols, nlags=NLAGS)
leaky_cols = ["leaked_target_"+str(i) for i in range(NLAGS)]
train = train.join(all_df.set_index("ID")[leaky_cols], on="ID", how="left")
test = test.join(all_df.set_index("ID")[leaky_cols], on="ID", how="left")

# Backward Fill
def fill(df):
    df = df.replace(0,np.nan)
    df = df.fillna(method='bfill', axis= 1)
    return df

# Time Series
y = train["target"].copy()
ts_cols = ["target"] + [col for col in train.columns if col.startswith('leaked_target')]
ts_cols = ts_cols[::-1]

# ReOrder
train = train[ts_cols]
test = test[ts_cols]

# Get Mean
train["nonzero_mean"] = train.apply(lambda x: np.expm1(np.log1p(x[x!=0]).mean()), axis=1)
test["nonzero_mean"] = test.apply(lambda x: np.expm1(np.log1p(x[x!=0]).mean()), axis=1)

# Fll Zero Values
lstm_train = fill(train)
lstm_test = fill(test)
# Log Scaling (MinMaxScaler probably doesn't play as well)
lstm_train = np.log1p(lstm_train)
lstm_test = np.log1p(lstm_test)
# Scaling
# target_scaler = MinMaxScaler()
# print(target_scaler.fit(lstm_train['target'].values.reshape(-1, 1)))
# lstm_train['target']= target_scaler.transform(lstm_train['target'].values.reshape(-1, 1))

y = lstm_train.target.copy()
lstm_train.drop("target", axis=1, inplace=True)
lstm_test.drop("target", axis=1, inplace=True)

# lstm_train_scaler = MinMaxScaler()
# lstm_train = pd.DataFrame(lstm_train_scaler.fit_transform(lstm_train), columns = lstm_train.columns)
# lstm_test_scaler = MinMaxScaler()
# lstm_test = pd.DataFrame(lstm_test_scaler.fit_transform(lstm_test), columns = lstm_test.columns)
# Save for use elsewhere
lstm_train.to_csv("Mohin_train.csv", index=True)
lstm_test.to_csv("Mohin_test.csv", index=True)
X = lstm_train.values.reshape(lstm_train.shape[0],lstm_train.shape[1],1)
test_df = lstm_test.values.reshape(lstm_test.shape[0],lstm_test.shape[1],1)
print("X Shape: ", X.shape)
print("Test DF: ", test_df.shape)
del lstm_train, lstm_test

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=1, shuffle=False)
# Utility
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

inputshape = (X.shape[1], X.shape[2])
print("Input Shape: ", inputshape)

LSTM_PARAM = {"batch_size":64,
              "verbose":2,
              "epochs":45}

# Model Architecture
model_lstm = Sequential([
    LSTM(100, input_shape=inputshape),
    Activation('relu'),
    Dropout(0.5),
    Dense(10),
    Activation('relu'),
    Dropout(0.5),
    Dense(1, activation = 'linear')
])

# Objective Function
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Compile
opt = optimizers.Adam()
model_lstm.compile(optimizer=opt, loss="mae")
callbacks_list=[EarlyStopping(monitor="val_loss",min_delta=.1, patience=3, mode='auto')]
hist = model_lstm.fit(X_train, y_train,
                      validation_data=(X_valid, y_valid),
                      callbacks=callbacks_list,
                      **LSTM_PARAM)

# Model Evaluation
best = np.argmin(hist.history["val_loss"])
print("Optimal Epoch: ",best+1)
print("Train Score: {}, Validation Score: {}".format(hist.history["loss"][best],hist.history["val_loss"][best]))

plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='validation')
plt.xlabel("Epochs")
plt.ylabel("Mean Square Error")
plt.title("Train and Validation Error")
plt.legend()
plt.savefig("Train and Validation MSE Progression.png")
plt.show()
# Transform
pred = model_lstm.predict(test_df)
pred = [x for sl in pred for x in sl]
# scaled = target_scaler.inverse_transform(np.array([x for sl in pred for x in sl]).reshape(-1, 1))
# scaled = [x for sl in scaled for x in sl]
display(pd.Series(y).describe())
display(pd.Series(pred).describe())
# Visualize
# y.hist(label="Training Ground Truth")
# pd.Series(pred).hist(label="Submission Prediction")
# f, ax = plt.subplots()
# sns.kdeplot(y, shade=True, color="r", ax = ax)
# sns.kdeplot(pred, shade=True, color="b", ax = ax)
# plt.legend(loc='upper right')
# plt.title("Distribution for Ground Truth and Submission")
# plt.show
# Submit
submission = pd.Series(np.expm1(pred))
submission.rename("target", inplace=True)
submission.index = testdex
submission.index.name = "ID"
submission.to_csv("LSTM_submission.csv",index=True,header=True)
submission.head()
