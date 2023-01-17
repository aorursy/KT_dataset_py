# Set environment variables

import os

import time

import random

import pickle

import warnings

import numpy as np

import pandas as pd

import lightgbm as lgb



VERSION = 1

BASE_PATH = f"/kaggle/working/m5-forecasting-accuracy-ver{VERSION}"
# Turn off warnings



warnings.filterwarnings("ignore")
# Seed everything



SEED = 9453

random.seed(SEED)

np.random.seed(SEED)
# Function to join dataframes



def join_dataframe(df1, df2, columns):

    df = df1.join(df2.set_index(columns), on = columns)

    return df
# Function to create lag and rolling lag features



def create_lag_features(df, day_list):



    columns = ["id", "d", "sales"]

    df_temp = df[columns]



    for day in day_list:

        if day <= 30:

            df_temp[f"sales_lag_{str(day)}"] = df_temp.groupby("id")["sales"].transform(lambda x: x.shift(day)).astype(np.float16)

        else:

            df_temp[f"sales_lag_{str(day)}_max"] = df_temp.groupby("id")["sales"].transform(lambda x: x.shift(28).rolling(day).max()).astype(np.float16)

            df_temp[f"sales_lag_{str(day)}_mean"] = df_temp.groupby("id")["sales"].transform(lambda x: x.shift(28).rolling(day).mean()).astype(np.float16)

            df_temp[f"sales_lag_{str(day)}_std"] = df_temp.groupby("id")["sales"].transform(lambda x: x.shift(28).rolling(day).std()).astype(np.float16)



    df_temp.drop(["sales"], axis = 1, inplace = True)

    df = join_dataframe(df, df_temp, ["id", "d"])

    return df
# Function to select features



def select_features(df, store_id, day_list):



    # Basic sales columns

    sales_columns = [

        "id"

        , "item_id"

        , "dept_id"

        , "cat_id"

        , "d"

        , "sales"

        , "available_after"

    ]



    # Calendar features

    state = store_id[:2]

    calendar_features = [

        "event_name_1"

        , "event_type_1"

        , "event_name_2"

        , "event_type_2"

        , f"snap_{state}"

        , "day"

        , "weekday"

        , "week"

        , "month"

        , "year"

        , "week_of_month"

        , "is_weekend"

    ]

    

    # Price features

    price_features = [

        "item_nunique"

        , "price_mean"

        , "price_std"

        , "price_mean_change_month"

        , "price_mean_change_year"

    ]



    # Lag features

    lag_features = [f"sales_lag_{str(day)}" for day in day_list if day <= 30]

    rolling_lag_features = [f"sales_lag_{str(day)}_{stats}" for day in day_list if day > 30 for stats in ["max", "mean", "std"]]



    # Collect columns

    columns = sales_columns + calendar_features + price_features + lag_features + rolling_lag_features



    # Specify categorical features, features and target variable

    categorical_features = ["item_id", "dept_id", "cat_id", "event_name_1", "event_type_1", "event_name_2", "event_type_2", f"snap_{state}"]

    features_before_encoded = [column for column in columns if column not in ["id", "d", "sales"]]

    target_variable = "sales"



    return df[columns], categorical_features, features_before_encoded, target_variable
# Function to encode categorical features



def mean_encoding(df, categorical_columns, target_variable):

    for column in categorical_columns:

        df[column + "_encoded"] = df.groupby(column)[target_variable].transform("mean").astype(np.float16)

    return df
# Function to load, pre-process datasets



def load_and_preprocess(store_id, version):



    # Set environment variables

    SALES_PATH = f"/kaggle/input/m5-forecasting-accuracy-sales-basic-features"

    CALENDAR_PRICE_PATH = f"/kaggle/input/m5-forecasting-accuracy-calendar-price-features"



    # Specify some variables

    state = store_id[:2]

    target_variable = "sales"

    day_list = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 60, 90, 180, 365]



    # Get sales basic features and filtered by store_id

    df_sales_features = pd.read_pickle(f"{SALES_PATH}/m5-forecasting-accuracy-ver{version}/sales_basic_features.pkl")

    df_sales_features = df_sales_features[df_sales_features["store_id"] == store_id]



    # Get calendar features

    df_calendar_features = pd.read_pickle(f"{CALENDAR_PRICE_PATH}/m5-forecasting-accuracy-ver{version}/calendar_features.pkl")

    df_raw = join_dataframe(df_sales_features, df_calendar_features, ["d"])



    # Get price features

    df_price_features = pd.read_pickle(f"{CALENDAR_PRICE_PATH}/m5-forecasting-accuracy-ver{version}/price_features.pkl")

    df_raw = join_dataframe(df_raw, df_price_features, ["store_id", "item_id", "d"])



    # Create lag features

    df_raw = create_lag_features(df_raw, day_list)



    # Select necessary features

    df_raw, categorical_features, features_before_encoded, target_variable = select_features(df_raw, store_id, day_list)



    # Encode categorical features

    df_raw = mean_encoding(df_raw, categorical_features, target_variable)

    features = [feature + "_encoded" if feature in categorical_features else feature for feature in features_before_encoded]



    return df_raw, features, target_variable
# Function to train



def train(store_id, lgb_params, version):



    # Record how much time it takes

    start_time = time.time()

    print(f"{store_id} starts!")

    print(f"Start preprocessing datasets for {store_id}")



    # Get data for the store_id

    df_raw, features, target_variable = load_and_preprocess(store_id, version)



    # Record how much time it takes

    preprocess_time = time.time()

    print(f"Preprocessing time: {round(preprocess_time - start_time)} seconds")

    print(f"Start splitting datasets for {store_id}")



    # Split train, validation and test datasets

    number_of_train = 1913

    number_of_validation = 56

    train_mask = df_raw["d"] <= number_of_train

    validation_mask = train_mask & (df_raw["d"] > (number_of_train - number_of_validation))

    test_mask = df_raw["d"] > number_of_train    



    # Save created datasets as bin

    # Check this issue: https://github.com/Microsoft/LightGBM/issues/1032

    # Save test dataset as pickle file

    df_train = lgb.Dataset(df_raw[train_mask][features], label = df_raw[train_mask][target_variable])

    df_validation = lgb.Dataset(df_raw[validation_mask][features], label = df_raw[validation_mask][target_variable], reference = df_train)

    df_test = df_raw[test_mask].reset_index(drop = True)

    try:

        os.remove("df_train.bin")

        os.remove("df_validation.bin")

        os.remove(f"df_test_{store_id}.pkl")

    except:

        pass

    df_train.save_binary("df_train.bin")

    df_validation.save_binary("df_validation.bin")

    df_test.to_pickle(f"df_test_{store_id}.pkl")



    # Reload from bin files

    df_train = lgb.Dataset("df_train.bin")

    df_validation = lgb.Dataset("df_validation.bin")



    # Record how much time it takes

    splitting_time = time.time()

    print(f"Splitting time: {round(splitting_time - preprocess_time)} seconds")

    print(f"Start training datasets for {store_id}")



    # Train models

    estimator = lgb.train(lgb_params, df_train, valid_sets = [df_validation], verbose_eval = 100)



    # Record how much time it takes

    training_time = time.time()

    print(f"Training time: {round(training_time - splitting_time)} seconds")

    print(f"{store_id} finished!")



    # Save models

    pickle.dump(estimator, open(f"model_{store_id}_{version}", "wb"))



    return features, target_variable
# Set training parameters

# Check: https://lightgbm.readthedocs.io/en/latest/Parameters.html#core-parameters



lgb_params = {

    "boosting": "gbdt" # There are 4 types: gbdt, rf, dart, goss

    # "goss" trains faster but relatively low performance, "dart" has higher accurary but taking too much time

    , "objective": "tweedie" # Tweedie is a distribution that is really worthy of having a deep understanding

    , "tweedie_variance_power": 1.5 # Default value

    # Closer to 2 means shifting towards a Gamma distribution, closer to 1 means shifting towards a Poisson distribution

    , "metric": ["rmse"]

    , "bagging_fraction": 0.5

    , "bagging_freq": 10

    , "learning_rate": 0.05 # Default is 0.1

    , "num_leaves": 2 ** 12 - 1 # Default is 31, make it higher to train precisely

    , "min_data_in_leaf": 2 ** 12 - 1 # Default is 20

    , "feature_fraction": 0.8 # Default is 1

    , "max_bin": "150" # Default is 255, smaller to prevent over-fitting

    , "num_iterations": 1000 # We set a large number in order to keep training going

    , "boost_from_average": False # Default is true, adjusting initial score to the mean of labels for faster convergence

    , "seed": SEED # Seed everything

    , "verbose": -1 # Not showing too much information while training

}
# Start training



store_id_list = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]

for store_id in store_id_list:

    model_path = f"/kaggle/input/m5-forecasting-accuracy-model-training-v3/model_{store_id}_{VERSION}"

    if not os.path.exists(model_path):

        features, target_variable = train(store_id, lgb_params, VERSION)
# Function to predict



def make_prediction(store_id, version):



    # Record how much time it takes

    start_time = time.time()

    print(f"{store_id} starts!")

    print(f"Start loading datasets and model for {store_id}")



    # Load datasets and model

    df_test = pd.read_pickle(f"/kaggle/input/m5-forecasting-accuracy-model-training-v3/df_test_{store_id}.pkl")

    estimator = pickle.load(open(f"/kaggle/input/m5-forecasting-accuracy-model-training-v3/model_{store_id}_{version}", "rb"))

    nonfeature_columns = ["id", "item_id", "dept_id", "cat_id", "d", "sales", "event_name_1", "event_type_1", "event_name_2", "event_type_2", f"snap_{store_id[:2]}"]

    features = [column for column in df_test.columns if column not in nonfeature_columns]

    target_variable = "sales"



    # Record how much time it takes

    loading_time = time.time()

    print(f"Loading time: {round(loading_time - start_time)} seconds")

    print(f"Start predicting for {store_id}")



    # Make prediction for last 28 days

    number_of_train = 1913

    number_of_prediction = 28

    df_store_prediction = pd.DataFrame()

    for i in range(1, number_of_prediction + 1):

        df_temp = df_test.copy()



        # Predict sales

        day_mask = df_temp["d"] == (number_of_train + i)

        df_temp[target_variable][day_mask] = estimator.predict(df_temp[day_mask][features])



        # Rename columns

        df_temp = df_temp[day_mask][["id", target_variable]]

        df_temp.columns = ["id", f"F{i}"]



        # Join predictions

        if "id" in list(df_store_prediction):

            df_store_prediction = df_store_prediction.merge(df_temp, on = ["id"], how = "left")

        else:

            df_store_prediction = df_temp.copy()



    # Record how much time it takes

    predicting_time = time.time()

    print(f"Predicting time: {round(predicting_time - loading_time)} seconds")

    print(f"{store_id} finished!")



    return df_store_prediction
print("I have canceled the execution here.")



# df_prediction = pd.DataFrame()

# for store_id in store_id_list:

#     df_temp = make_prediction(store_id, VERSION)

#     if "id" in list(df_prediction):

#         df_prediction = pd.concat([df_prediction, df_temp])

#     else:

#         df_prediction = df_temp.copy()

# df_prediction = df_prediction.reset_index(drop = True)

# df_prediction.to_pickle(f"df_prediction_{VERSION}")

# df_prediction