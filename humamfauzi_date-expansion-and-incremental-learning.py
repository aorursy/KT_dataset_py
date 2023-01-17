# Initial Python environment setup...
import numpy as np # linear algebra
import pandas as pd # CSV file I/O (e.g. pd.read_csv)
import os # reading the input files we have access to
import time
import gc

import sys

# Multiprocessing
# from multiprocessing import Pool

# Import Sklearn and lgbm
import lightgbm as lgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

try:
    import cPickle as pickle
except BaseException:
    import pickle

#Visualization
files = os.listdir('../input')
print(files)
# %%time
# Test Unit

# training_step = list(range(0, 20_000_001, 1_000_000))
# model = None
# origin_cols = ['key',
#  'fare_amount',
#  'pickup_datetime',
#  'pickup_longitude',
#  'pickup_latitude',
#  'dropoff_longitude',
#  'dropoff_latitude',
#  'passenger_count']
# print(training_step)

# for i in range(1, len(training_step)):
#     start = time.time()
#     files = os.listdir('../')
#     print(files)
#     if "model.pkl" not in files:
#         print("No Model yet...")
#         model = None
#     else:
#         with open('../model.pkl', 'rb') as fin:
#             model = pickle.load(fin)
            
#     data = pd.read_csv("../input/train.csv", skiprows=training_step[i-1], nrows=training_step[i])
#     data.columns = origin_cols
#     model = incremental_learning(data, model)
#     print("TRAINING NUMBER:", i,"Elapsed:", time.time() - start, "s\n")
    
#     del data
#     with open('../model.pkl', 'wb') as fout:
#         pickle.dump(model, fout)
#     del model
    

# test_df = pd.read_csv("../input/train.csv", skiprows=40_000_000, nrows=1_000_000)
# test_df.columns = origin_cols

# add_travel_vector_features(test_df)
# test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"], infer_datetime_format=True)
# test_X = pd.concat([test_df, timeExpansion(test_df["pickup_datetime"])], axis=1)

# test_y_true = test_X["fare_amount"]
# test_X.drop(["key", "pickup_datetime", "fare_amount"], axis=1, inplace=True)

# print("Test Shape: ",test_X.shape)

# test_y_pred = model.predict(test_X)
# print ("TEST MAE:", mean_absolute_error(test_y_true, test_y_pred))
# Main Pipeline
def incremental_learning(data, model):
    start = time.time()
    
    data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"], infer_datetime_format=True)
    print("Conversion took", time.time() - start, "s")
    add_travel_vector_features(data)
    data = pd.concat([data, timeExpansion(data["pickup_datetime"])], axis=1)
    
    
    
    X_tr, X_te, y_tr, y_te = train_test_split(
        data.drop(["key", "pickup_datetime", "fare_amount"], axis=1), 
        data['fare_amount'], 
        test_size=0.2)
    
    del data
    
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression_l1',
        'metric': {'l1'},
        'num_leaves': 12,
        'learning_rate': 0.1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'random_state': 77,
        'n_jobs': 4,
        'bagging_freq': 5,
        'verbose': 2
    }

    
    dset = lgbm.Dataset(X_tr, y_tr, free_raw_data=True)
    model = lgbm.train(params, 
                       init_model = model,
                       train_set = dset,
                       keep_training_booster = True,
                       num_boost_round = 100)
    MAE = mean_absolute_error(y_te, model.predict(X_te))
    print("Mean Absolute Error:", MAE)
    return model
# Helper Function

# def PandasMultiprocessing(Series, function, pool):
#     start = time.time()
#     n_cores = 4
#     lin = np.linspace(0, len(Series), n_cores+1, dtype="int")
    
#     print("Start Multiprocessing")
    
#     result = pool.starmap_async(function, [[Series[lin[num-1]:lin[num]]] for num in range(1, len(lin))])    
    
#     converted = pd.concat(result.get())
    
#     print("Multiprocessing Took", time.time() - start, "s")
#     return converted

def partial_import(filename, skip, rows):
    return pd.read_csv('..input/train.csv', skiprows=skip, nrows=rows)

def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()
    
def month_translation(number):
    month_list = ["jan", "feb", "mar", 
                   "apr", "may", "jun", "jul", 
                   "aug", "sep", "oct", "nov", "dec"]
    
    for num, i in enumerate(month_list):
        if num == number-1:
            return i

def quarter_translation(number):
    quarter_border = [0, 3, 6, 9, 12, 15, 18, 21, 24]
    quarter_list = ["11", "12", "13", "14", "21", "22", "23", "24"]
    for num, i in enumerate(range(1, len(quarter_border))):
        if (number < quarter_border[i]) & (number >= quarter_border[i-1]):
            return quarter_list[num]

def week_translation(number):
    week_border = [0, 7, 14, 21, 31]
    week_list = ["1W", "2W", "3W", "4W"]
    for num, i in enumerate(range(1, len(week_border))):
         if (number <= week_border[i]) & (number > week_border[i-1]):
            return week_list[num]


# add year, 12 month, week, and 8 quarter time status
def timeExpansion(timeSeries):
    additional_cols = ["year", "month", "quarter", "week"]
    dummy0 = pd.DataFrame(np.zeros((len(timeSeries), len(additional_cols)), 'int'), 
                         columns=additional_cols, 
                         index=timeSeries.index)
    
    dummy0["year"] = [i.year for i in timeSeries]
    dummy0["month"] = [month_translation(i.month) for i in timeSeries]
    dummy0["day"] = [week_translation(i.day) for i in timeSeries]
    dummy0["quarter"] = [quarter_translation(i.hour) for i in timeSeries]
    return pd.get_dummies(dummy0)

def predict_and_submit(submit=True):
    if "model.pkl" in files:
        with open('../model.pkl', 'rb') as fin:
            estimator = pickle.load(fin)
    
    
    test_df = pd.read_csv('../input/test.csv')
    add_travel_vector_features(test_df)
    test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"], infer_datetime_format=True)
    test_X = pd.concat([test_df, timeExpansion(test_df["pickup_datetime"])], axis=1)
    test_X.drop(["key", "pickup_datetime"], axis=1, inplace=True)

    print("Test Shape: ",test_X.shape)

    # Predict fare_amount on the test set using our model (w) trained on the training set.
    test_y_predictions = estimator.predict(test_X)

    # Write the predictions to a CSV file which we can submit to the competition.
    if submit:
        submission = pd.DataFrame(
            {'key': test_df.key, 'fare_amount': test_y_predictions},
            columns = ['key', 'fare_amount'])
        submission.to_csv('submission.csv', index = False)
#MAIN Execution
training_step = list(np.arange(40_000_000, 55_000_001, 1_000_000))
estimator = None
origin_cols = ['key',
 'fare_amount',
 'pickup_datetime',
 'pickup_longitude',
 'pickup_latitude',
 'dropoff_longitude',
 'dropoff_latitude',
 'passenger_count']

for i in range(1, len(training_step)):
    start = time.time()
    files = os.listdir("../")
    if "model.pkl" in files:
        with open('../model.pkl', 'rb') as fin:
            estimator = pickle.load(fin)
    
    print(training_step[i-1], training_step[i])
    train_df = pd.read_csv("../input/train.csv", skiprows=training_step[i-1], nrows=training_step[i])
    train_df.columns = origin_cols
    
    estimator = incremental_learning(train_df, estimator)
    
    print("Operation", i, "took ", time.time() - start, " \n")
    del train_df
    with open('../model.pkl', 'wb') as fout:
        pickle.dump(estimator, fout)
    del estimator
    gc.collect()
    
predict_and_submit()