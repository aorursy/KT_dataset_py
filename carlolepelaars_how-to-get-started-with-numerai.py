# Install Numerai's API

!pip install numerapi

# Get the latest version of Weights and Biases

!pip install wandb --upgrade
# Obfuscated WANDB API Key

from kaggle_secrets import UserSecretsClient

WANDB_KEY = UserSecretsClient().get_secret("WANDB_API_KEY")
import os

import numpy as np

import random as rn

import pandas as pd

import seaborn as sns

import lightgbm as lgb

import matplotlib.pyplot as plt

from scipy.stats import spearmanr

from sklearn.metrics import mean_absolute_error



# Initialize Numerai's API

import numerapi

NAPI = numerapi.NumerAPI(verbosity="info")



# Weights and Biases

import wandb

from wandb.lightgbm import wandb_callback

wandb.login(key=WANDB_KEY)



# Data directory

DIR = "/kaggle/working"



# Set seed for reproducability

seed = 1234

rn.seed(seed)

np.random.seed(seed)

os.environ['PYTHONHASHSEED'] = str(seed)



# Surpress Pandas warnings

pd.set_option('chained_assignment', None)
def download_current_data(directory: str):

    """

    Downloads the data for the current round

    :param directory: The path to the directory where the data needs to be saved

    """

    current_round = NAPI.get_current_round()

    if os.path.isdir(f'{directory}/numerai_dataset_{current_round}/'):

        print(f"You already have the newest data! Current round is: {current_round}")

    else:

        print(f"Downloading new data for round: {current_round}!")

        NAPI.download_current_dataset(dest_path=directory, unzip=True)



def load_data(directory: str, reduce_memory: bool=True) -> tuple:

    """

    Get data for current round

    :param directory: The path to the directory where the data needs to be saved

    :return: A tuple containing the datasets

    """

    print('Loading the data')

    full_path = f'{directory}/numerai_dataset_{NAPI.get_current_round()}/'

    train_path = full_path + 'numerai_training_data.csv'

    test_path = full_path + 'numerai_tournament_data.csv'

    train = pd.read_csv(train_path)

    test = pd.read_csv(test_path)

    

    # Reduce all features to 32-bit floats

    if reduce_memory:

        num_features = [f for f in train.columns if f.startswith("feature")]

        train[num_features] = train[num_features].astype(np.float32)

        test[num_features] = test[num_features].astype(np.float32)

        

    val = test[test['data_type'] == 'validation']

    return train, val, test
# Download, unzip and load data

download_current_data(DIR)

train, val, test = load_data(DIR, reduce_memory=True)
print("Training data:")

display(train.head(2))

print("Test data:")

display(test.head(2))
print("Training set info:")

train.info()
print("Test set info:")

test.info()
# Extract era numbers

train["erano"] = train.era.str.slice(3).astype(int)

plt.figure(figsize=[14, 6])

train.groupby(train['erano'])["target_kazutsugi"].size().plot(title="Era sizes", figsize=(14, 8));
feats = [f for f in train.columns if "feature" in f]

plt.figure(figsize=(15, 5))

sns.distplot(pd.DataFrame(train[feats].std()), bins=100)

sns.distplot(pd.DataFrame(val[feats].std()), bins=100)

sns.distplot(pd.DataFrame(test[feats].std()), bins=100)

plt.legend(["Train", "Val", "Test"], fontsize=20)

plt.title("Standard deviations over all features in the data", weight='bold', fontsize=20);
def sharpe_ratio(corrs: pd.Series) -> np.float32:

    """

    Calculate the Sharpe ratio for Numerai by using grouped per-era data

    

    :param corrs: A Pandas Series containing the Spearman correlations for each era

    :return: A float denoting the Sharpe ratio of your predictions.

    """

    return corrs.mean() / corrs.std()





def evaluate(df: pd.DataFrame) -> tuple:

    """

    Evaluate and display relevant metrics for Numerai 

    

    :param df: A Pandas DataFrame containing the columns "era", "target_kazutsugi" and a column for predictions

    :param pred_col: The column where the predictions are stored

    :return: A tuple of float containing the metrics

    """

    def _score(sub_df: pd.DataFrame) -> np.float32:

        """Calculates Spearman correlation"""

        return spearmanr(sub_df["target_kazutsugi"], sub_df["prediction_kazutsugi"])[0]

    

    # Calculate metrics

    corrs = df.groupby("era").apply(_score)

    payout_raw = (corrs / 0.2).clip(-1, 1)

    spearman = round(corrs.mean(), 4)

    payout = round(payout_raw.mean(), 4)

    numerai_sharpe = round(sharpe_ratio(corrs), 4)

    mae = mean_absolute_error(df["target_kazutsugi"], df["prediction_kazutsugi"]).round(4)



    # Display metrics

    print(f"Spearman Correlation: {spearman}")

    print(f"Average Payout: {payout}")

    print(f"Sharpe Ratio: {numerai_sharpe}")

    print(f"Mean Absolute Error (MAE): {mae}")

    return spearman, payout, numerai_sharpe, mae
def get_group_stats(df: pd.DataFrame) -> pd.DataFrame:

    """

    Create features by calculating statistical moments for each group.

    

    :param df: Pandas DataFrame containing all features

    """

    for group in ["intelligence", "wisdom", "charisma", "dexterity", "strength", "constitution"]:

        cols = [col for col in df.columns if group in col]

        df[f"feature_{group}_mean"] = df[cols].mean(axis=1)

        df[f"feature_{group}_std"] = df[cols].std(axis=1)

        df[f"feature_{group}_skew"] = df[cols].skew(axis=1)

    return df
# Add group statistics features

train = get_group_stats(train)

val = get_group_stats(val)

test = get_group_stats(test)
# Calculate correlations with target

full_corr = train.corr()

corr_with_target = full_corr["target_kazutsugi"].T.apply(abs).sort_values(ascending=False)



# Select features with highest correlation to the target variable

features = corr_with_target[:150]

features.drop("target_kazutsugi", inplace=True)
print("Top 10 Features according to correlation with target:")

features[:10]
# Create list of most correlated features

feature_list = features.index.tolist()
# Configuration for hyperparameter sweep

sweep_config = {

   'method': 'grid',

   'metric': {

          'name': 'mse',

          'goal': 'minimize'   

        },

   'parameters': {

       "num_leaves": {'values': [30, 40, 50]}, 

       "max_depth": {'values': [4, 5, 6, 7]}, 

       "learning_rate": {'values': [0.1, 0.05, 0.01]},

       "bagging_freq": {'values': [7]}, 

       "bagging_fraction": {'values': [0.6, 0.7, 0.8]}, 

       "feature_fraction": {'values': [0.85, 0.75, 0.65]},

   }

}

sweep_id = wandb.sweep(sweep_config, project="numerai_tutorial")
# Prepare data for LightGBM

dtrain = lgb.Dataset(train[feature_list], label=train["target_kazutsugi"])

dvalid = lgb.Dataset(val[feature_list], label=val["target_kazutsugi"])

watchlist = [dtrain, dvalid]



def _train():

    # Configure and train model

    wandb.init(name="LightGBM_sweep")

    lgbm_config = {"num_leaves": wandb.config.num_leaves, "max_depth": wandb.config.max_depth, "learning_rate": wandb.config.learning_rate,

                   "bagging_freq": wandb.config.bagging_freq, "bagging_fraction": wandb.config.bagging_fraction, "feature_fraction": wandb.config.feature_fraction,

                   "metric": 'mse', "random_state": seed}

    lgbm_model = lgb.train(lgbm_config, train_set=dtrain, num_boost_round=750, valid_sets=watchlist, 

                           callbacks=[wandb_callback()], verbose_eval=100, early_stopping_rounds=50)

    

    # Create predictions for evaluation

    val_preds = lgbm_model.predict(val[feature_list], num_iteration=lgbm_model.best_iteration)

    val.loc[:, "prediction_kazutsugi"] = val_preds

    # W&B log metrics

    spearman, payout, numerai_sharpe, mae = evaluate(val)

    wandb.log({"Spearman": spearman, "Payout": payout, "Numerai Sharpe Ratio": numerai_sharpe, "Mean Absolute Error": mae})

    

# Run hyperparameter sweep (grid search)

wandb.agent(sweep_id, function=_train)
# Train model with best configuration

wandb.init(project="numerai_tutorial", name="LightGBM")

best_config = {"num_leaves": 50, "max_depth": 6, "learning_rate": 0.1,

               "bagging_freq": 7, "bagging_fraction": 0.6, "feature_fraction": 0.75,

               "metric": 'mse', "random_state": seed}

lgbm_model = lgb.train(best_config, train_set=dtrain, num_boost_round=750, valid_sets=watchlist, 

                       callbacks=[wandb_callback()], verbose_eval=100, early_stopping_rounds=50)

    

# Create final predictions from best model

train.loc[:, "prediction_kazutsugi"] = lgbm_model.predict(train[feature_list], num_iteration=lgbm_model.best_iteration)

val.loc[:, "prediction_kazutsugi"] = lgbm_model.predict(val[feature_list], num_iteration=lgbm_model.best_iteration)
# Evaluate Model

print("--- Final Training Scores ---")

spearman, payout, numerai_sharpe, mae = evaluate(train)

print("\n--- Final Validation Scores ---")

spearman, payout, numerai_sharpe, mae = evaluate(val)
# Calculate feature exposure

all_features = [col for col in train.columns if 'feature' in col]

feature_spearman_val = [spearmanr(val["prediction_kazutsugi"], val[f])[0] for f in all_features]

feature_exposure_val = np.std(feature_spearman_val).round(4)
print(f"Feature exposure on validation set: {feature_exposure_val}")
# Set API Keys for submitting to Numerai

PUBLIC_ID = "YOUR PUBLIC ID"

SECRET_KEY = "YOUR SECRET KEY"



# Initialize API with API Keys

napi = numerapi.NumerAPI(public_id=PUBLIC_ID, 

                          secret_key=SECRET_KEY, 

                          verbosity="info")

# Upload predictions for current round

test.loc[:, "prediction_kazutsugi"] = lgbm_model.predict(test[feature_list], num_iteration=lgbm_model.best_iteration)

test[['id', "prediction_kazutsugi"]].to_csv("submission.csv", index=False)
# Upload predictions to Numerai

# napi.upload_predictions("submission.csv", tournament=napi.get_current_round())
print("Submission File:")

test[['id', "prediction_kazutsugi"]].head(2)