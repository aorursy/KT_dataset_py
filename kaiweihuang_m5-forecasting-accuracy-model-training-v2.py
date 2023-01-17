# Set environment variables

import os

import random

import warnings

import numpy as np

import pandas as pd

import lightgbm as lgb

from IPython.display import HTML, display



VERSION = 1

BASE_PATH = f"/kaggle/working/m5-forecasting-accuracy-ver{VERSION}"
# Turn off warnings



warnings.filterwarnings("ignore")
# Seed everything



SEED = 9453

random.seed(SEED)

np.random.seed(SEED)
# Memory usage function and join function



def format_memory_usage(total_bytes):

    unit_list = ["", "Ki", "Mi", "Gi"]

    for unit in unit_list:

        if total_bytes < 1024:

            return f"{total_bytes:.2f}{unit}B"

        total_bytes /= 1024

    return f"{total_bytes:.2f}{unit}B"



def join_dataframe(df1, df2, columns):

    df = df1.join(df2.set_index(columns), on = columns)

    return df
# Function to import basic features



def load_basic_dataset_with_downsampling(number_of_slice, version):



    # Set environment variables

    SALES_PATH = f"/kaggle/input/m5-forecasting-accuracy-sales-basic-features"

    CALENDAR_PRICE_PATH = f"/kaggle/input/m5-forecasting-accuracy-calendar-price-features"



    # Get sales basic features and downsampling

    df_sales_features = pd.read_pickle(f"{SALES_PATH}/m5-forecasting-accuracy-ver{version}/sales_basic_features.pkl")

    ids = np.array_split(list(df_sales_features["id"].unique()), number_of_slice)[0]

    df_sales_features = df_sales_features[df_sales_features["id"].isin(ids)].reset_index(drop = True)



    # Get calendar features

    df_calendar_features = pd.read_pickle(f"{CALENDAR_PRICE_PATH}/m5-forecasting-accuracy-ver{version}/calendar_features.pkl")

    calendar_selected_columns = df_calendar_features.columns.tolist()

    calendar_selected_columns.remove("date")

    df_features = join_dataframe(df_sales_features, df_calendar_features[calendar_selected_columns], ["d"])



    # Get price features

    df_price_features = pd.read_pickle(f"{CALENDAR_PRICE_PATH}/m5-forecasting-accuracy-ver{version}/price_features.pkl")

    df_features = join_dataframe(df_features, df_price_features, ["store_id", "item_id", "d"])



    return df_features
# Set global variables



NUMBER_OF_SLICE = 10 # 10% downsampling

NUMBER_OF_TRAIN = 1913 # Not to include test data
df_fast_basic_training = load_basic_dataset_with_downsampling(NUMBER_OF_SLICE, VERSION)

df_fast_basic_training = df_fast_basic_training[df_fast_basic_training["d"] <= NUMBER_OF_TRAIN].reset_index(drop = True)

df_fast_basic_training.info()
# Set training parameters

# Check: https://lightgbm.readthedocs.io/en/latest/Parameters.html#core-parameters



lgb_params = {

    "boosting": "gbdt" # There are 4 types: gbdt, rf, dart, goss

    , "objective": "regression" # RMSE

    , "metric": ["rmse"]

    , "bagging_fraction": 0.5

    , "bagging_freq": 10

    , "learning_rate": 0.05 # Default is 0.1

    , "num_leaves": 2 ** 8 - 1 # Default is 31

    , "min_data_in_leaf": 2 ** 8 - 1 # Default is 20

    , "feature_fraction": 0.8 # Default is 1

    , "num_iterations": 10000 # We set a large number in order to keep training going

    , "early_stopping_round": 20 # Default is 0, and we will stop the training if it stops improving

    , "seed": SEED # Seed everything

    , "verbose": -1 # Not showing too much information while training

}
# Gather feature list



target_feature = "sales"

remove_feature_list = ["id", "d", target_feature]

feature_list = [column for column in list(df_fast_basic_training) if column not in remove_feature_list]

print(feature_list)
# Function to make fast training test



def make_fast_training(df, feature_list, number_of_train, lgb_params):



    # Get actual features

    feature_list = [column for column in list(df) if column not in remove_feature_list]



    # Set aside 28 days for validation

    train_X, train_y = df[df["d"] <= (number_of_train - 28)][feature_list], df[df["d"] <= (number_of_train - 28)][target_feature]

    validation_X, validation_y = df[df["d"] > (number_of_train - 28)][feature_list], df[df["d"] > (number_of_train - 28)][target_feature]

    train_data = lgb.Dataset(train_X, label = train_y)

    validation_data = lgb.Dataset(validation_X, label = validation_y)



    # Train

    # Check: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html

    estimator = lgb.train(lgb_params, train_data, valid_sets = [train_data ,validation_data], verbose_eval = 500)



    return estimator
# Train baseline model



baseline_model = make_fast_training(df_fast_basic_training, feature_list, NUMBER_OF_TRAIN, lgb_params)
# Function to import rolling lag features



def load_rolling_features(df, version):



    # Set environment variables

    SALES_ROLLING_PATH = f"/kaggle/input/m5-forecasting-accuracy-sales-rolling-features"



    # Get rolling lag features

    df_rolling_features = pd.read_pickle(f"{SALES_ROLLING_PATH}/m5-forecasting-accuracy-ver{version}/sales_rolling_features.pkl")

    rolling_selected_columns = df_rolling_features.columns.tolist()

    rolling_selected_columns.remove("sales")

    df_features = join_dataframe(df, df_rolling_features[rolling_selected_columns], ["id", "d"])



    return df_features
df_fast_rolling_training = load_rolling_features(df_fast_basic_training, VERSION)

df_fast_rolling_training.info()
# Gather feature list



feature_list = [column for column in list(df_fast_rolling_training) if column not in remove_feature_list]

print(feature_list)
# Train rolling lag model



rolling_model = make_fast_training(df_fast_rolling_training, feature_list, NUMBER_OF_TRAIN, lgb_params)
# Feature importance of baseline model



lgb.plot_importance(baseline_model, figsize = (12, 8))
# Feature importance of lag model



lgb.plot_importance(rolling_model, figsize = (18, 12))
# Function to calculate RMSE



def rmse(y, y_prediction):

    return np.sqrt(np.mean(np.square(y - y_prediction)))
# Get validation datasets



feature_list = [column for column in list(df_fast_rolling_training) if column not in remove_feature_list]

df_validation = df_fast_rolling_training[df_fast_rolling_training["d"] > (NUMBER_OF_TRAIN - 28)].reset_index(drop = True)

df_validation.head(10)
# Make predictions and calculate RMSE base



df_validation["prediction"] = rolling_model.predict(df_validation[feature_list])

base_rmse = rmse(df_validation[target_feature], df_validation["prediction"])

print(f"Base RMSE: {base_rmse}")
# Calculate permutation importance



for feature in feature_list:

    df_temp = df_validation.copy()

    if df_temp[feature].dtypes.name != "category":

        df_temp[feature] = np.random.permutation(df_temp[feature].values)

        df_temp["prediction"] = rolling_model.predict(df_temp[feature_list])

        permuted_rmse = rmse(df_temp[target_feature], df_temp["prediction"])

        rmse_difference = np.round(permuted_rmse - base_rmse, 4)

        print(f"The RMSE difference after permuted feature {feature}: {rmse_difference}")
# Import PCA and set parameters

from sklearn.decomposition import PCA



N_COMPONENTS = 10
# Gather only numeric features



numeric_feature_list = []

for feature in feature_list:

    if df_fast_rolling_training[feature].dtypes.name != "category":

        numeric_feature_list.append(feature)

print(numeric_feature_list)
# Beaware: PCA cannot accept NaN and infinite values



df_fast_rolling_training_pca = df_fast_rolling_training[numeric_feature_list].fillna(0).replace([np.inf, -np.inf], 0)
# Run PCA



pca = PCA(n_components = N_COMPONENTS, random_state = SEED)

pca.fit(df_fast_rolling_training_pca)

rolling_after_pca = pca.transform(df_fast_rolling_training_pca)
# Print PCA results



print(sum(pca.explained_variance_ratio_))

print(pca.explained_variance_ratio_)

print(pca.singular_values_)
# Check the components



df_pca = pd.DataFrame(

    pca.components_

    , columns = df_fast_rolling_training_pca.columns

    , index = [f"PCA-{i}" for i in range(1, 11)]

)

display(HTML(df_pca.to_html()))