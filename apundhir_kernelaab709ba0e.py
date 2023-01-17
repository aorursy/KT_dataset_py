import datetime

import math

import os



import numpy as np

import pandas as pd



from joblib import Parallel, delayed

from lightgbm import LGBMRegressor, LGBMClassifier

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error

from tqdm import tnrange, tqdm_notebook
DATA_PATH = "../input/"

TEST_ROWS_NUMBER = 93

FOLD_COUNT = 10

SKIP_FOLD_COUNT = 1

N_JOBS = 4
train_data = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))

test_data = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))



weather = pd.read_csv(os.path.join(DATA_PATH, "weather.csv"))

temperatures = pd.read_csv(os.path.join(DATA_PATH, "temperatures.csv"))

landmarks = pd.read_csv(os.path.join(DATA_PATH, "landmarks.csv"))

hexagon_centers = pd.read_csv(os.path.join(DATA_PATH, "hexagon_centers.csv"))

public_holidays = pd.read_csv(os.path.join(DATA_PATH, "public_holidays.csv"))
def parse_date(date):

    return datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
hexagon_centers_dict = {}



for row in hexagon_centers.values:

    hexagon_centers_dict[row[0]] = (row[1], row[2])
def compute_statistics(data):

    return data.describe().transpose().drop(columns=["count", "min"])
def slice_historical_data(date, min_lag, max_lag):

    date = parse_date(date)

    

    min_lag_delta = datetime.timedelta(days=min_lag)

    max_lag_delta = datetime.timedelta(days=max_lag)

    

    result = []

    for row in train_data.values:

        row_date = parse_date(row[0])

        

        if date - max_lag_delta <= row_date <= date - min_lag_delta:

            result.append(row)



    result = pd.DataFrame(data=result, columns=train_data.columns)

    return result
### On same day in history



def slice_same_day(date):

    date = parse_date(date)

    

    result = []

    for row in train_data.values:

        row_date = parse_date(row[0])

        if (date.weekday() - row_date.weekday()) == 0:

            result.append(row)

            

    result = pd.DataFrame(data=result, columns=train_data.columns)

    return result
def slice_same_hr(date):

    date = parse_date(date)

    

    result = []

    for row in train_data.values:

        row_date = parse_date(row[0])

        if (date.hour - row_date.hour) == 0:

            result.append(row)

            

    result = pd.DataFrame(data=result, columns=train_data.columns)

    return result
def find_nearest_hexagons(hexagon_id, k):

    x, y = hexagon_centers_dict[hexagon_id]

    distances = []

    

    for another_hexagon_id in hexagon_centers_dict:

        if another_hexagon_id == hexagon_id:

            continue



        another_x, another_y = hexagon_centers_dict[another_hexagon_id]

        distance = (x - another_x) ** 2 + (y - another_y) ** 2

        

        distances.append((distance, another_hexagon_id))

    

    distances = sorted(distances)

    return [another_hexagon_id for _, another_hexagon_id in distances[:k]]





def build_features_for_hexagon(hexagon_id, date):

    historical_data = slice_historical_data(date, min_lag=30, max_lag=60)

    return historical_data
def build_features_for_date(date):

    features = []



    historical_data = slice_historical_data(date, min_lag=30, max_lag=60)

    statistics = compute_statistics(historical_data.drop(columns="Date"))

    

    same_day_history = slice_same_day(date)

    statistics_2 = compute_statistics(same_day_history.drop(columns="Date"))

    

    same_hr_history = slice_same_hr(date)

    statistics_3 = compute_statistics(same_hr_history.drop(columns="Date"))

    

    

    lag_30_60_features = pd.DataFrame(

        statistics.values,

        columns=[

            "mean_lag_30_60",

            "std_lag_30_60",

            "25_per_lag_30_60",

            "50_per_lag_30_60",

            "75_per_lag_30_60",

            "max_lag_30_60"

        ]

    )

    

    same_day_features = pd.DataFrame(

        statistics_2.values,

        columns=[

            "mean_same_day",

            "std_same_day",

            "25_per_same_day",

            "50_per_same_day",

            "75_per_same_day",

            "max_lag_same_day"

        ]

    )

    

    same_hr_features = pd.DataFrame(

        statistics_3.values,

        columns=[

            "mean_same_hr",

            "std_same_hr",

            "25_per_same_hr",

            "50_per_same_hr",

            "75_per_same_hr",

            "max_lag_same_hr"

        ]

    )

    

    

    features.append(lag_30_60_features)

    features.append(same_day_features)

    features.append(same_hr_features)

    

    return pd.concat(features, axis=1)
def build_features(data):

    result = []



    parallel = Parallel(n_jobs=N_JOBS, backend="multiprocessing")

    result = parallel(delayed(build_features_for_date)(date) for date in data.values[:, 0])



    if not result:

        return None

    return pd.concat(result)



def get_targets(data):

    result = []

    for row in data.values:

        for target in row[1:]:

            result.append(target)



    return np.array(result)

def split_cross_validation(train_data, fold_id):

    rows_by_fold = len(train_data) // FOLD_COUNT

    

    rows_for_train = rows_by_fold * (fold_id + 1) - TEST_ROWS_NUMBER



    train = train_data[:rows_for_train]

    val = train_data[rows_for_train: rows_for_train + TEST_ROWS_NUMBER]



    return train, val
def extract_features_and_targets(data):

    val_features = build_features(data)

    val_targets = get_targets(data)

    return val_features, val_targets
def build_submission(test_predictions):

    sample_submission = pd.read_csv(os.path.join(DATA_PATH, "sampleSubmission.csv"))

    index = 0

    

    result = []

    for date in test_data.values[:, 0]:

        for hex_id in test_data.columns[1:]:

            id = "{}_{}".format(date, hex_id)

            result.append([id, test_predictions[index]])

            index += 1

    

    return pd.DataFrame(data=result, columns=["Id", "Incidents"])
folds_features = []

folds_targets = []



for fold_id in tnrange(SKIP_FOLD_COUNT, FOLD_COUNT):

    _, val = split_cross_validation(train_data, fold_id)

    features, targets = extract_features_and_targets(val)

    

    if features is None:

        continue

    

    folds_features.append(features)

    folds_targets.append(targets)

test_features, _ = extract_features_and_targets(test_data)
class Model():

    def __init__(self):

        self._impl = LinearRegression()



    def fit(self, X, y):

        self._impl.fit(X, y)



    def predict(self, X):

        return self._impl.predict(X)

models = []

scores = []

test_predictions = []



for fold_id in range(len(folds_features)):

    model = Model()

    

    val_features = folds_features[fold_id]

    val_targets = folds_targets[fold_id]

    

    train_features = pd.concat(folds_features[:fold_id] + folds_features[fold_id + 1:])

    train_targets = np.concatenate(folds_targets[:fold_id] + folds_targets[fold_id + 1:])

    

    model.fit(train_features, train_targets)



    val_predicts = model.predict(val_features)



    test_predictions.append(model.predict(test_features))

    models.append(model)



    rmse = mean_squared_error(val_predicts, val_targets) ** 0.5

    print(fold_id, rmse)

    scores.append(rmse)

np.mean(scores)
submission = build_submission(np.mean(test_predictions, axis=0))
submission.to_csv("submit1.csv", index=False)
folds_features[0].head()
def baseline_model():

    # create model

    model = Sequential()

    model.add(Dense(12, input_dim=13, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal'))

    # Compile model

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model