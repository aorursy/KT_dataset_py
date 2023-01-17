# https://github.com/stared/livelossplot

!pip install livelossplot tensorflow-gpu
!nvidia-smi
import numpy as np

import pandas as pd

from datetime import datetime

from pandas import DataFrame

from typing import List, NamedTuple, Tuple

import seaborn as sns



from IPython.display import display

from pathlib import Path

from matplotlib import pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from livelossplot.keras import PlotLossesCallback



pd.set_option('display.max_rows', 100)

pd.set_option('display.max_columns', 50)

pd.set_option('display.width', 1000)



ROOT_DIR = Path("/kaggle/input/bike-sharing-demand")

TRAIN_DATA_PATH = ROOT_DIR / "train.csv"

TEST_DATA_PATH = ROOT_DIR / "test.csv"
def expanded_index_datetime_col(data: DataFrame) -> DataFrame:

    data = data.copy()

    data["hour"] = data.index.hour

    data["weekday"] = data.index.weekday

    data["month"] = data.index.month

    data["year"] = data.index.year

    return data



def replaced_with_onehot_cols(data: DataFrame, col_names: List[str]) -> DataFrame:

    data = data.copy()

    

    for col_name in col_names:

        one_hot = pd.get_dummies(data[col_name], prefix=col_name)

        data = data.join(one_hot)

        

        # Original column is not needed anymore

        del data[col_name]

    return data
def load(path: Path) -> DataFrame:

    return pd.read_csv(path, parse_dates=True, index_col="datetime")



def correlation(df: DataFrame) -> DataFrame:

    corr = df.corr()

    return sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)



def prepare(df: DataFrame) -> DataFrame:

    df = df.copy()

    df = expanded_index_datetime_col(df)

    df = replaced_with_onehot_cols(df, col_names=["season", "holiday", "workingday", "weather", "weekday", "month", "year"])

    df = df.drop(["casual", "registered", "atemp"], axis=1, errors="ignore")

    return df



original_train: DataFrame = load(TRAIN_DATA_PATH)

display(original_train.describe())

display(correlation(original_train))



train_val: DataFrame = prepare(original_train)
def normalize_cols(df: DataFrame, scaler) -> DataFrame:

    df = df.copy()

    return DataFrame(scaler.fit_transform(df.values), columns=df.columns, index=df.index)



x_scaler = MinMaxScaler()

x_trainval = train_val.drop("count", axis=1)

x_trainval = normalize_cols(df=x_trainval, scaler=x_scaler)



y_scaler = MinMaxScaler()

y_trainval = train_val[["count"]]

y_trainval = normalize_cols(df=y_trainval, scaler=y_scaler)



x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=0.1)

_, NUM_FEATURES = x_train.shape
from tensorflow.keras.layers import Input, Dense, Dropout

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import tensorflow.keras.backend as K

import tensorflow as tf
def rmsle_K(y, y_hat):

    return K.sqrt(K.mean(K.square(tf.math.log1p(y) - tf.math.log1p(y_hat))))



def small_model():

    input = Input(shape=(NUM_FEATURES, ))

    _ = Dense(8, activation='relu')(input)

    output = Dense(1, activation='relu')(_)

    return Model(inputs=input, outputs=output)



def medium_model():

    input = Input(shape=(NUM_FEATURES, ))

    _ = Dense(32, activation='relu')(input)

    _ = Dropout(0.4)(_)

    _ = Dense(32, activation='relu')(_)

    _ = Dropout(0.4)(_)

    _ = Dense(16, activation='relu')(_)

    output = Dense(1, activation='relu')(_)

    return Model(inputs=input, outputs=output)



def large_model():

    """Far too many parameters for this problem."""

    input = Input(shape=(NUM_FEATURES, ))

    _ = Dense(64, activation='relu')(input)

    _ = Dropout(0.5)(_)

    _ = Dense(64, activation='relu')(_)

    _ = Dropout(0.5)(_)

    _ = Dense(64, activation='relu')(_)

    _ = Dropout(0.5)(_)

    _ = Dense(64, activation='relu')(_)

    output = Dense(1, activation='relu')(_)

    return Model(inputs=input, outputs=output)
def compile(model: Model) -> Model:

    model.compile(optimizer='adam', loss=rmsle_K, metrics=['mse'])

    return model



class TrainingResult(NamedTuple):

    model: Model

    train_loss: float

    val_loss: float



def train(model, x: np.ndarray, y: np.ndarray, 

          val_data=Tuple[np.ndarray, np.ndarray]) -> TrainingResult:

    history = model.fit(x, y,

                    validation_data=val_data,

                    epochs=200, 

                    batch_size=64,

                    verbose=1, 

                    callbacks=[

                        PlotLossesCallback(), 

                        ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=5, min_lr=0.000001, verbose=1),

                        EarlyStopping(monitor="val_loss", patience=10, verbose=1),

                    ])

    return TrainingResult(model, 

                          train_loss=history.history['loss'][-1], 

                          val_loss=history.history['val_loss'][-1])



models = {

    "small": small_model(),

    "medium": medium_model(),

    "large": large_model()

}
results = {}

for name, model in models.items():

    result = train(compile(model), x_train.values, y_train.values, val_data=(x_val.values, y_val.values))

    results[name] = result
for name, result in results.items():

    print(name, result.train_loss, result.val_loss)
def show_predictions(model: Model, x: DataFrame, y: DataFrame, title="preds"):

    norm_preds = model.predict(x.values)

    predictions = y_scaler.inverse_transform(norm_preds)

    targets = y_scaler.inverse_transform(y[["count"]])

    

    x = x.sort_index()

    fig = plt.figure(figsize=(25,9))

    plt.title(title, fontsize=24)

    plt.plot(x.index, targets, label="target")

    plt.plot(x.index, predictions, alpha=0.7, label="prediction")

    plt.legend()

    plt.show()





s = slice(400, 500)

model = results["small"].model

# model = results["medium"].model

# model = results["large"].model

show_predictions(model, x_train[s], y_train[s], title="Predictions on train set")

show_predictions(model, x_val[s], y_val[s], title="Predictions on validation set")
def evaluate(df: DataFrame, normalize=True) -> np.array:

    if normalize:

        x_test = x_scaler.fit_transform(df.values)

    else:

        x_test = df.values

    preds = model.predict(x_test)

    if normalize:

        return y_scaler.inverse_transform(preds)

    return preds



def save_submission(test_df: DataFrame, preds: np.array, path: str):

    submission = test_df.copy()

    submission["datetime"] = test_df.index

    submission["count"] = preds

    submission = submission[["datetime", "count"]]

    submission.to_csv(path, index=False)





original_test: DataFrame = load(TEST_DATA_PATH)

test: DataFrame = prepare(original_test)

print(list(x_train.columns))

print(list(test.columns))

assert list(x_train.columns) == list(test.columns)

test = normalize_cols(df=test, scaler=x_scaler)



predictions = evaluate(test)

save_submission(test, predictions, path=f"/kaggle/working/v13.csv")



fig = plt.figure(figsize=(25,9))

plt.title("Predictions on test set", fontsize=24)

plt.plot(test.index, predictions)

plt.show()
from sklearn.linear_model import LinearRegression

from sklearn import tree

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor



regressors = {

    "Linear": LinearRegression(),

    "SVR": SVR(),

    "XGB": XGBRegressor(),

    "RandomForest": RandomForestRegressor(),

    "NN": results["large"].model

}



def evaluate_model(model, x: DataFrame) -> np.ndarray:

    norm_preds = model.predict(x.values)

    predictions = y_scaler.inverse_transform(norm_preds.reshape(-1, 1))

    return predictions

    

def plot(xs, y_predictions, y_targets=None):

    plt.plot(xs, y_predictions, alpha=0.5, color='r', label='prediction')

    if y_targets is not None:

        plt.plot(xs, y_targets, color='b', label="target")

    plt.legend()

    plt.show()



s = slice(1000, 1200)



for name, regressor in regressors.items():

    if name == "NN":

        regr_model = regressor

    else:

        regr_model = regressor.fit(x_train.values, y_train.values)    

    

    # Validation set

    predictions = evaluate_model(regr_model, x_val[s])

    targets = y_scaler.inverse_transform(y_val[s][["count"]])

    

    fig = plt.figure(figsize=(25,9))

    plt.title(name, fontsize=24)

    plot(x_val[s].sort_index().index, predictions, targets)

    

    # Test set

    predictions = evaluate_model(regr_model, test)

    plot(test.index, predictions)

    save_submission(test, predictions, path=f"/kaggle/working/{name}_v13.csv")