import numpy as np # linear algebra

import pandas as pd # data processing,

# Loading data

_df = pd.read_csv("/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv")



_SydneyData = _df[_df["Location"] == "Sydney"]

_SydneyData
_SydneyData.count().sort_values()
# get rid of unnecesary columns

_SydneyData = _SydneyData.drop(columns=["RISK_MM", "Date", "Location"])

_SydneyData.shape
# get rid of missing values

_SydneyData = _SydneyData.dropna(how="any")

_SydneyData.shape
# detect the outliers using Z-Score

from scipy import stats

_z = np.abs(stats.zscore(_SydneyData._get_numeric_data()))

print(_z)

_SydneyData = _SydneyData[(_z < 3).all(axis=1)]

print(_SydneyData.shape)
# translate yes/no to binary 1/0.

_SydneyData["RainToday"].replace({"No": 0, "Yes": 1}, inplace = True)

_SydneyData["RainTomorrow"].replace({"No": 0, "Yes": 1}, inplace = True)

# convert unique value to one-hot-representation

_categorical_columns = ["WindGustDir", "WindDir3pm", "WindDir9am"]

for col in _categorical_columns:

    print(np.unique(_SydneyData[col]))



_SydneyData = pd.get_dummies(_SydneyData, columns = _categorical_columns)

_SydneyData
# standardize data

from sklearn import preprocessing

_scaler = preprocessing.MinMaxScaler()

_scaler.fit(_SydneyData)

_SydneyData = pd.DataFrame(_scaler.transform(_SydneyData), index = _SydneyData.index, columns = _SydneyData.columns)

_SydneyData
# Feature selection

from sklearn.feature_selection import SelectKBest, chi2

_x = _SydneyData.loc[:, _SydneyData.columns != "RainTomorrow"]

_y = _SydneyData[["RainTomorrow"]]

_selector = SelectKBest(chi2, k = 5)

_selector.fit(_x, _y)

_x_new = _selector.transform(_x)

print(_x.columns[_selector.get_support(indices = True)])
# _SydneyData = _SydneyData[["Cloud3pm", "RainToday", "WindDir3pm_SSW", "RainTomorrow"]]

# 3 features

# _Data = _SydneyData[["Cloud3pm", "RainToday", "WindDir3pm_SSW"]]

# all features

# _Data = _SydneyData.loc[:, _SydneyData.columns != "RainTomorrow"]

# 5 features

# _Data = _SydneyData[["Rainfall", "Sunshine", "Cloud3pm", "RainToday", "WindDir3pm_SSW"]]



_Data = _x_new

_Label = _SydneyData["RainTomorrow"]



_Data.shape
from keras.optimizers import Adam

from keras.layers import Dense, Activation, Input, BatchNormalization

from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split

from keras.models import Model

from keras.regularizers import l2



_TrainData, _TestData, _TrainLabel, _TestLabel = train_test_split(_Data, _Label, test_size = 0.25)



_i = Input(shape = (_Data.shape[1],))

_x = Dense(64, kernel_regularizer = l2(0.005))(_i)

_x = BatchNormalization()(_x)

_x = Activation("relu")(_x)

_x = Dense(64, kernel_regularizer = l2(0.005))(_x)

_x = BatchNormalization()(_x)

_x = Activation("relu")(_x)

_o = Dense(1, activation = "sigmoid")(_x)



_model = Model(_i, _o)

_model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

_callback = EarlyStopping(monitor='val_loss', patience=500, verbose=1, mode='auto')

_model.fit(_TrainData, _TrainLabel, validation_data = (_TestData, _TestLabel), batch_size = 32, epochs = 10000, callbacks = [_callback])

_, _train_acc = _model.evaluate(_TrainData, _TrainLabel, verbose=0)

_, _test_acc = _model.evaluate(_TestData, _TestLabel, verbose=0)

print('Train: %.3f, Test: %.3f' % (_train_acc, _test_acc))