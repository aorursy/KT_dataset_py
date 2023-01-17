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
!pip install keras-tuner
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import models, layers

import kerastuner as kt
from kerastuner import HyperModel
pd.plotting.register_matplotlib_converters()
train_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/train.csv")
test_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/test.csv")
train_df = train_df.copy()
test_df = test_df.copy()
train_df.info()
test_df.tail()
train_df = train_df.drop(["Id", "Population", "Weight",
                         "Province_State", "County"], axis=1)
test_df = test_df.drop(["ForecastId", "Population", "Weight",
                       "Province_State", "County"], axis=1)
LE = LabelEncoder()

train_df["Country_Region"] = LE.fit_transform(train_df["Country_Region"])
train_df["Date"] = LE.fit_transform(train_df["Date"])
train_df["Target"] = LE.fit_transform(train_df["Target"])

test_df["Country_Region"] = LE.fit_transform(test_df["Country_Region"])
test_df["Date"] = LE.fit_transform(test_df["Date"])
test_df["Target"] = LE.fit_transform(test_df["Target"])

train_df.tail()
HotEnc = OneHotEncoder(sparse=True)

train_df["Country_Region"] = HotEnc.fit_transform(
    train_df["Country_Region"].values.reshape(-1, 1)).toarray()
train_df["Date"] = HotEnc.fit_transform(
    train_df["Date"].values.reshape(-1, 1)).toarray()

test_df["Country_Region"] = HotEnc.fit_transform(
    test_df["Country_Region"].values.reshape(-1, 1)).toarray()
test_df["Date"]= HotEnc.fit_transform(
    test_df["Date"].values.reshape(-1, 1)).toarray()

test_df.head()
x_train = train_df.iloc[:500000, :-1]
y_train = train_df.iloc[:500000, -1]

x_test = train_df.iloc[500001:, :-1]
y_test = train_df.iloc[500001:, -1]

x_test.head()
SC = StandardScaler()

x_train = SC.fit_transform(x_train)
x_test = SC.fit_transform(x_test)
batch_size = 10000
model = models.Sequential([
    tf.keras.layers.Dense(32, activation="relu", 
                 input_shape=(x_train.shape[1],)), 
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1)])

model.compile(optimizer="RMSprop", loss="mse", metrics=["mse"])
    
history = model.fit(x_train, y_train, epochs=3,
                    batch_size=batch_size)

model.evaluate(x_test, y_test)
class RegressionHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        
    def build(self, hp):
        model = models.Sequential()
        model.add(
            layers.Dense(units=hp.Int(
                "units", 8, 64, 4, default=8),
                activation=hp.Choice(
                "dense_activation",
                values=["relu", "tanh", "sigmoid"],
                default="relu"),
                input_shape=input_shape))
        model.add(layers.Dense(units=hp.Int(
            "units", 16, 64, 4, default=16),
            activation=hp.Choice(
                "dense_activation",
                values=["relu", "tanh", "sigmoid"],
            default="relu")))
        model.add(layers.Dropout(hp.Float(
            "dropout", min_value=0.0, max_value=0.1,
             default=0.005, step=0.01)))
        model.add(layers.Dense(1))
        
        model.compile(optimizer="RMSprop", loss="mse",
                      metrics=["mse"])
        
        return model
input_shape = (x_train.shape[1],)

hypermodel = RegressionHyperModel(input_shape)
tuner_bayesian = kt.BayesianOptimization(hypermodel, 
                                         objective="mse",
                                         max_trials=10,
                                         seed=13,
                                         executions_per_trial=2)
tuner_bayesian.search(x_train, y_train, epochs=2, 
                      validation_split=0.2, verbose=0)

best_model = tuner_bayesian.get_best_models(num_models=1)[0]
best_model
best_model.evaluate(x_test, y_test)
best_model.summary()
prediction = best_model.predict(test_df).flatten()
predictions
test_df["Target"] = LE.inverse_transform(test_df["Target"])

sub = pd.DataFrame({"Target": test_df["Target"],
                   "Predictions": prediction})

sub.head()