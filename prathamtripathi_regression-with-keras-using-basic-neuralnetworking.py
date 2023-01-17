import pandas as pd

import numpy as np
import keras

from keras.layers import Dense

from keras.models import Sequential
df = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")

df.head()
df.isnull().sum()
df_cols = df.columns
predictor = df[df_cols[df_cols != "quality"]]

target = df[["quality"]]
predictor_normal = ((predictor - predictor.mean())/predictor.std())

predictor_normal.head()
n_cols = predictor_normal.shape[1]
def regression_model():

    model = Sequential()

    model.add(Dense(50,activation = "relu", input_shape =(n_cols,)))

    model.add(Dense(50,activation = "relu"))

    model.add(Dense(50,activation = "relu"))

    

    model.compile(optimizer = "adam", loss = "mean_squared_error")

    return model
model = regression_model()

model.fit(predictor_normal,target, validation_split = 0.3, epochs = 100, verbose = 1)