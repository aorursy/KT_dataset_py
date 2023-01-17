import pandas as pd

import numpy as np

import keras

from keras.models import Sequential

from keras.layers import Dense
df = pd.read_csv("../input/regression-with-neural-networking/concrete_data.csv")

df.head()
df.isnull().sum()
df_columns = df.columns
pred = df[df_columns[df_columns!="Strength"]]

targ = df["Strength"]
targ.head()
pred_normal = ((pred - pred.mean())/pred.std())

pred_normal.head()
n_cols = pred_normal.shape[1]
def regression_model():

    model = Sequential()

    model.add(Dense(50,activation = "relu",input_shape = (n_cols,)))

    model.add(Dense(50,activation ="relu"))

    model.add(Dense(50,activation ="relu"))

    model.compile(optimizer = "adam", loss = "mean_squared_error")

    

    return model
model = regression_model()

model.fit(pred_normal,targ,validation_split = 0.3, epochs = 100, verbose = 1)