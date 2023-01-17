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
data_train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv", parse_dates = ["Date"])
data_test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv", parse_dates = ["Date"])
submission = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/submission.csv")
data_train.Province_State = data_train.Province_State.fillna("N")
data_test.Province_State = data_test.Province_State.fillna("N")
interaction = data_train["Province_State"] + "_" + data_train["Country_Region"]
interaction_test = data_test["Province_State"] + "_" + data_test["Country_Region"]
data_train["Interaction"] = interaction
data_test["Interaction"] = interaction_test
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
cl_case = MinMaxScaler()
cl_fate = MinMaxScaler()
le = LabelEncoder()
data_train["ConfirmedCases_norm"] = 0
data_train["Fatalities_norm"] = 0
data_train["ConfirmedCases_norm"] = cl_case.fit_transform(data_train["ConfirmedCases"].values.reshape(-1,1))
data_train["Fatalities_norm"] = cl_fate.fit_transform(data_train["Fatalities_norm"].values.reshape(-1,1))
data_train["CR_label"] = le.fit_transform(data_train.Interaction)
data_train.head()
#creating windows
def create_window(interval, prediction_day,data, column):
    X = []
    y = []
    groupby_Interaction = data.groupby("Interaction")
    Interaction_list = data.Interaction.unique()
    for Interaction_ind in Interaction_list:
        county = groupby_Interaction.get_group(Interaction_ind)
        county = county[["ConfirmedCases", "Fatalities", "CR_label"]]
        for i in range(county.shape[0]-interval-prediction_day):
            X.append(county.iloc[i:i+interval].values)
            y.append(county.iloc[i+interval:i+interval+prediction_day, column].values)
    return np.array(X), np.array(y)
interval = 5
prediction_day = 1
data_case_X, data_case_y = create_window(interval, prediction_day, data_train, 0)
data_f_X, data_f_y = create_window(interval, prediction_day, data_train, 1)
data_case_y.shape
from sklearn.model_selection import train_test_split
X_c_train, X_c_test, y_c_train, y_c_test = train_test_split(data_case_X, data_case_y, test_size=0.33, random_state=42)
X_f_train, X_f_test, y_f_train, y_f_test = train_test_split(data_f_X, data_f_y, test_size=0.33, random_state=42)
X_c_train = X_c_train.reshape((X_c_train.shape[0],X_c_train.shape[1],3))
X_c_test = X_c_test.reshape((X_c_test.shape[0],X_c_test.shape[1],3))
X_f_train = X_f_train.reshape((X_f_train.shape[0],X_f_train.shape[1],3))
X_f_test = X_f_test.reshape((X_f_test.shape[0],X_f_test.shape[1],3))
from keras.losses import MeanSquaredLogarithmicError
from keras import Sequential
from keras import layers
from keras import Input
from keras.utils import plot_model
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
model_c = Sequential()
model_c.add(layers.LSTM(130,input_shape=(interval,3)))
model_c.add(layers.Dense(65))
model_c.add(layers.Dropout(rate = 0.2))
model_c.add(layers.Dense(32))
model_c.add(layers.Dense(prediction_day))

model_f = Sequential()
model_f.add(layers.LSTM(130,input_shape=(interval,3)))
model_f.add(layers.Dense(65))
model_f.add(layers.Dropout(rate = 0.2))
model_f.add(layers.Dense(32))
model_f.add(layers.Dense(prediction_day))
callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=4, verbose=1, factor=0.6),
             EarlyStopping(monitor='val_loss', patience=20),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
model_c.compile(loss=[MeanSquaredLogarithmicError()], optimizer="adam")
model_f.compile(loss=[MeanSquaredLogarithmicError()], optimizer="adam")
history_c = model_c.fit(X_c_train, y_c_train, 
          epochs = 100, 
          batch_size = 1000, 
          validation_data=(X_c_test,  y_c_test), 
          callbacks=callbacks)
history_f = model_f.fit(X_f_train, y_f_train, 
          epochs = 100, 
          batch_size = 1000, 
          validation_data=(X_f_test,  y_f_test), 
          callbacks=callbacks)
plt.plot(history_c.history['loss'])
plt.plot(history_c.history['val_loss'])

plt.plot(history_f.history['loss'])
plt.plot(history_f.history['val_loss'])
country_ = data_test.Interaction.unique()
#first 7 days
predict_c = []
predict_f = []
old_cases = data_train.groupby('Interaction').get_group("N_Afghanistan")
test_df = old_cases[["ConfirmedCases", "Fatalities", "CR_label"]]
for repeat in range(1):
    trial = test_df.iloc[-interval:]
    input_x = trial.values.reshape(1,interval,3)
    pd_value_c= model_c.predict(input_x).reshape(1)
    pd_value_f= model_f.predict(input_x).reshape(1) 
    predict_c.extend(pd_value_c)
    predict_f.extend(pd_value_f)
    new_df = pd.DataFrame({"ConfirmedCases":pd_value_c, "Fatalities": pd_value_f})
    new_df["CR_label"] = trial.iloc[-1,2]
    test_df = pd.concat([test_df, new_df], axis = 0)


country_ = data_test.Interaction.unique()
Cp = []
Fp = []
for ct in country_:
    predict_c = []
    predict_f = []
    old_cases = data_train.groupby('Interaction').get_group(ct)
    test_df = old_cases[["ConfirmedCases", "Fatalities",  "CR_label"]]
    for repeat in range(44):
        trial = test_df.iloc[-interval:]
        input_x = trial.values.reshape(1,interval,3)
        pd_value_c= model_c.predict(input_x).reshape(1)
        pd_value_f= model_f.predict(input_x).reshape(1)
        
        predict_c.extend(pd_value_c)
        predict_f.extend(pd_value_f)
        
        new_df = pd.DataFrame({"ConfirmedCases":pd_value_c, "Fatalities": pd_value_f})
        new_df["CR_label"] = trial.iloc[-1,2]
        test_df = pd.concat([test_df, new_df], axis = 0)
    Cp.extend(predict_c[0:43])
    Fp.extend(predict_f[0:43])
submission["ConfirmedCases"] = Cp
submission["Fatalities"] = Fp
submission.head()
submission.to_csv("submission.csv",index=False)