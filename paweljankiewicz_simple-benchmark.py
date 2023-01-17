%matplotlib inline

import pandas as pd

from statsmodels.tsa.ar_model import AR

from statsmodels.tsa.arima_model import ARIMA

from tqdm import tqdm

from sklearn.metrics import mean_squared_error

import numpy as np



def root_mean_squared_error(t,p):

    return np.sqrt(mean_squared_error(t,p))
train_raw = pd.read_csv("../input/train.csv", parse_dates=["Date"])

train_raw.set_index("Date", inplace=True)

test = pd.read_csv("../input/test.csv", parse_dates=["Date"])

test.set_index("Date", inplace=True)

sample_submission = pd.read_csv("../input/sampleSubmission.csv")

import datetime
n_hexagons = 319
train_raw.shape
train_raw.head()
test.shape
train_raw.index.min(), train_raw.index.max()
test.index.min(), test.index.max()
pd.Series(train_raw.sum(axis=0).values).hist(bins=50)
train_raw["hex_299"].plot()
train_raw.iloc[:,1:].sum(axis=0).plot()
#function to convert a dataframe with predictions to submission format

def predictions_to_submission(predictions):

    records = []

    for hex_id in tqdm(range(0, n_hexagons)):

        hex_id = "hex_%03d" % hex_id

        for i,dt in enumerate(predictions.index):

            records.append((str(dt) + "_" + hex_id, predictions[hex_id][i]))

    submission = pd.DataFrame.from_records(records, columns=sample_submission.columns)

    return submission
split_date = datetime.datetime(2018,10,31)

train = train_raw[train_raw.index < split_date]

val_true = train_raw[train_raw.index >= split_date]

val_pred = val_true.copy()

val_pred.iloc[:,:] = 0



for hex_id in tqdm(range(n_hexagons)):

    hex_id = "hex_%03d" % hex_id

    data = train[hex_id].values

    val_pred[hex_id] = data.mean()

    

print("RMSE", root_mean_squared_error(val_true, val_pred))



# prepare final test

for hex_id in tqdm(range(n_hexagons)):

    hex_id = "hex_%03d" % hex_id

    data = train_raw[hex_id].values

    test[hex_id] = data.mean()



submission = predictions_to_submission(test)

submission.to_csv("submission_means.csv", index=False)
split_date = datetime.datetime(2018,10,31)

train = train_raw[train_raw.index < split_date]

val_true = train_raw[train_raw.index >= split_date]

val_pred = val_true.copy()

val_pred.iloc[:,:] = 0



for hex_id in tqdm(range(n_hexagons)):

    hex_id = "hex_%03d" % hex_id

    data = train[hex_id].values

    model = AR(data, )

    model_fit = model.fit()

    yhat = model_fit.predict(len(data), len(data)+93-1)

    val_pred[hex_id] = yhat

    

print("RMSE", root_mean_squared_error(val_true, val_pred))



for hex_id in tqdm(range(0, n_hexagons)):

    hex_id = "hex_%03d" % hex_id

    data = train_raw[hex_id].values

    model = AR(data)

    model_fit = model.fit()

    yhat = model_fit.predict(len(data), len(data)+94-1)

    test[hex_id] = yhat



submission = predictions_to_submission(test)

submission.to_csv("submission_autoreg.csv", index=False)