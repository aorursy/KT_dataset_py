import datetime

import math

import os



import numpy as np

import pandas as pd



from joblib import Parallel, delayed

from lightgbm import LGBMRegressor

from sklearn.metrics import mean_squared_error

from tqdm import tnrange, tqdm_notebook
DATA_PATH = "../input/"

TEST_ROWS_NUMBER = 93

FOLD_COUNT = 20

SKIP_FOLD_COUNT = 10

N_JOBS = 4
train_data = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))

test_data = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))



weather = pd.read_csv(os.path.join(DATA_PATH, "weather.csv"))

temperatures = pd.read_csv(os.path.join(DATA_PATH, "temperatures.csv"))

landmarks = pd.read_csv(os.path.join(DATA_PATH, "landmarks.csv"))

hexagon_centers = pd.read_csv(os.path.join(DATA_PATH, "hexagon_centers.csv"))

public_holidays = pd.read_csv(os.path.join(DATA_PATH, "public_holidays.csv"))
hexagon_to_landmarks = {}



for row in landmarks.values:

    hex_id = row[0]

    if hex_id[-3] == "_":

        hex_id = "{}_0{}".format(hex_id[:-3], hex_id[-2:])

    hexagon_to_landmarks[hex_id] = pd.DataFrame([row[1:]], columns=list(landmarks.columns)[1:])
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





def build_features_for_hexagons(data, postfix=""):

    mean_values = train_data.drop(columns=["Date"]).mean().to_dict()

    max_values = train_data.drop(columns=["Date"]).max().to_dict()

    std_values = train_data.drop(columns=["Date"]).std().to_dict()





    features = []

    for hexagon_id in data.columns[1:]:

        hexagon_features = []

        columns = []

        for k in [4, 10, 30]:

            nearest = find_nearest_hexagons(hexagon_id, k)

            nearest_mean = [mean_values[hex_id] for hex_id in nearest]

            nearest_max = [max_values[hex_id] for hex_id in nearest]

            nearest_std = [std_values[hex_id] for hex_id in nearest]

            

            for array, array_name in [(nearest_mean, "mean"), (nearest_max, "max"), (nearest_std, "std")]:

                for func, func_name in [(np.mean, "mean"), (np.max, "max"), (np.min, "min"), (np.std, "std")]:

                    hexagon_features.append(func(array))

                    columns.append("nearest_statistics_{}_{}_{}_{}".format(k, array_name, func_name, postfix))



        hexagon_features += list(hexagon_to_landmarks[hexagon_id].values[0])

        columns += list(hexagon_to_landmarks[hexagon_id].columns)

        features.append(hexagon_features)



    return pd.DataFrame(np.array(features), columns=columns)
def build_features_for_date(date):

    features = []



    ################################################################

    

    min_lag = 30

    for lag in [3, 7, 30, 90, 180]:

        historical_data = slice_historical_data(date, min_lag=min_lag, max_lag=min_lag + lag)

        

        ################################################################

        

        statistics = compute_statistics(historical_data.drop(columns="Date"))



        lag_features = pd.DataFrame(

            statistics.values,

            columns=[

                "mean_lag_{}_{}".format(min_lag, lag),

                "std_lag_{}_{}".format(min_lag, lag),

                "25_per_lag_{}_{}".format(min_lag, lag),

                "50_per_lag_{}_{}".format(min_lag, lag),

                "75_per_lag_{}_{}".format(min_lag, lag),

                "max_lag_{}_{}".format(min_lag, lag),

            ]

        )



        features.append(lag_features)

        

        ################################################################



        features.append(build_features_for_hexagons(historical_data, postfix="lag_{}_{}".format(min_lag, lag)))

        

        ################################################################

        

    

    ################################################################



    parsed_date = parse_date(date)

    weekday = parsed_date.weekday()

    hour = parsed_date.hour

    

    features.append(pd.DataFrame([[weekday, hour]] * len(features[0]), columns=["weeday", "hour"]))



    ################################################################

    

    columns = []

    

    for df in features:

        columns += list(df.columns)

    

    features = np.concatenate([df.values for df in features], axis=1)

    return pd.DataFrame(features, columns=columns)
def build_features(data):

    result = []



    result = []

    parallel = Parallel(n_jobs=N_JOBS, backend="multiprocessing", verbose=10)

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

    val_features = val_features.loc[:,~val_features.columns.duplicated()]

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

        self._impl = LGBMRegressor(learning_rate=0.01, n_estimators=1000, num_leaves=7)



    def fit(self, X, y):

        self._impl.fit(X, y)



    def predict(self, X):

        y = self._impl.predict(X)

        return y
COLUMNS_TO_USE = list(test_features.columns)
models = []

scores = []

test_predictions = []



for fold_id in range(len(folds_features)):

    model = Model()

    

    val_features = folds_features[fold_id][COLUMNS_TO_USE]

    val_targets = folds_targets[fold_id]

    

    train_features = pd.concat(folds_features[:fold_id] + folds_features[fold_id + 1:])[COLUMNS_TO_USE]

    train_targets = np.concatenate(folds_targets[:fold_id] + folds_targets[fold_id + 1:])

    

    model.fit(train_features, train_targets)



    val_predicts = model.predict(val_features)



    test_predictions.append(model.predict(test_features[COLUMNS_TO_USE]))

    models.append(model)



    rmse = mean_squared_error(val_predicts, val_targets) ** 0.5

    print(fold_id, rmse)

    scores.append(rmse)
np.mean(scores)
submission = build_submission(np.mean(test_predictions, axis=0))
submission.to_csv("submit.csv", index=False)