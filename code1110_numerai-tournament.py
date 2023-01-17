!pip install numerapi

import numerapi
import numpy as np

import pandas as pd

import os, sys

import gc

import pathlib

from typing import List, NoReturn, Union, Tuple, Optional, Text, Generic, Callable, Dict

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, QuantileTransformer

from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, mean_squared_error, mean_absolute_error, f1_score

from scipy.stats import spearmanr



from catboost import CatBoostRegressor, CatBoostClassifier

import lightgbm as lgb

import xgboost as xgb

import operator



# visualize

import matplotlib.pyplot as plt

import matplotlib.style as style

import seaborn as sns

from matplotlib import pyplot

from matplotlib.ticker import ScalarFormatter

sns.set_context("talk")

style.use('seaborn-colorblind')



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def get_int(x):

    try:

        return int(x[3:])

    except:

        return 1000

    

def read_data(data='train'):

    # get data 

    if data == 'train':

        df = pd.read_csv('https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_training_data.csv.xz')

    elif data == 'test':

        df = pd.read_csv('https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_tournament_data.csv.xz')

    

    # features

    feature_cols = df.columns[df.columns.str.startswith('feature')]

    

    # map to int, to reduce the memory demand

    mapping = {0.0 : 0, 0.25 : 1, 0.5 : 2, 0.75 : 3, 1.0 : 4}

    for c in feature_cols:

        df[c] = df[c].map(mapping).astype(np.uint8)

        

    # also cast era to int

    df["era"] = df["era"].apply(get_int)

    return df
%%time



# load train　(半年間固定)

train = read_data('train')

print(train.shape)

train.head()
%%time



# load test (毎週Roundごとに更新)

test = read_data('test')
valid = test[test["data_type"] == "validation"].reset_index(drop = True)



# validation split

valid.loc[valid["era"] > 180, "valid2"] = True # むずいやつ

valid.loc[valid["era"] <= 180, "valid2"] = False # 簡単なやつ
# remove data_type to save memory

train.drop(columns=["data_type"], inplace=True)

valid.drop(columns=["data_type"], inplace=True)

test.drop(columns=["data_type"], inplace=True)



print('The number of records: train {:,}, valid {:,}, test {:,}'.format(train.shape[0], valid.shape[0], test.shape[0]))
# features

features = [f for f in train.columns.values.tolist() if 'feature' in f]

print('There are {} features.'.format(len(features)))
target = 'target_kazutsugi' # いずれ正規分布のtarget_nomiになるらしい
# # create a model and fit (公式example)

# model = xgb.XGBRegressor(max_depth=5, learning_rate=0.01, n_estimators=2000, n_jobs=-1, colsample_bytree=0.1)

# model.fit(train[features], train[target])
%%time



# create a model and fit（LGBのハイパラは↑の公式XGBに寄せてみました）

params = {

            'n_estimators': 2000,

            'objective': 'regression',

            'boosting_type': 'gbdt',

            'max_depth': 5,

            'learning_rate': 0.01, 

            'feature_fraction': 0.1,

            'seed': 42

            }    

model = lgb.LGBMRegressor(**params)

model.fit(train[features], train[target])
pd.DataFrame(model.feature_importances_, index=features, columns=['importance']).sort_values(by='importance', ascending=False).style.background_gradient(cmap='viridis')
# naming conventions

PREDICTION_NAME = 'prediction'

TARGET_NAME = 'target'

# EXAMPLE_PRED = 'example_prediction'



# ---------------------------

# Functions

# ---------------------------

def valid4score(valid : pd.DataFrame, pred : np.ndarray) -> pd.DataFrame:

    """

    Generate new valid pandas dataframe for computing scores

    

    :INPUT:

    - valid : pd.DataFrame extracted from tournament data (data_type='validation')

    

    """

    valid_df = valid.copy()

    valid_df['prediction'] = pd.Series(pred).rank(pct=True, method="first")

    valid_df.rename(columns={target: 'target'}, inplace=True)

    

    return valid_df



def compute_corr(valid_df : pd.DataFrame):

    """

    Compute rank correlation

    

    :INPUT:

    - valid_df : pd.DataFrame where at least 2 columns ('prediction' & 'target') exist

    

    """

    

    return np.corrcoef(valid_df["target"], valid_df['prediction'])[0, 1]



def compute_max_drawdown(validation_correlations : pd.Series):

    """

    Compute max drawdown

    

    :INPUT:

    - validation_correaltions : pd.Series

    """

    

    rolling_max = (validation_correlations + 1).cumprod().rolling(window=100, min_periods=1).max()

    daily_value = (validation_correlations + 1).cumprod()

    max_drawdown = -(rolling_max - daily_value).max()

    

    return max_drawdown



def compute_val_corr(valid_df : pd.DataFrame):

    """

    Compute rank correlation for valid periods

    

    :INPUT:

    - valid_df : pd.DataFrame where at least 2 columns ('prediction' & 'target') exist

    """

    

    # all validation

    correlation = compute_corr(valid_df)

    print("ALL VALID: rank corr = {:.4f}".format(correlation))



    # first valid eras

    idx = np.where(valid_df["valid2"] == False)[0]

    correlation = compute_corr(valid_df.iloc[idx])

    print("VALID 1: rank corr = {:.4f}".format(correlation))



    # second valid eras

    idx = np.where(valid_df["valid2"] == True)[0]

    correlation = compute_corr(valid_df.iloc[idx])

    print("VALID 2: rank corr = {:.4f}".format(correlation))

    

def compute_val_sharpe(valid_df : pd.DataFrame):

    """

    Compute sharpe ratio for valid periods

    

    :INPUT:

    - valid_df : pd.DataFrame where at least 2 columns ('prediction' & 'target') exist

    """

    # all validation

    d = valid_df.groupby('era')[['target', 'prediction']].corr().iloc[0::2,-1].reset_index()

    me = d['prediction'].mean()

    sd = d['prediction'].std()

    max_drawdown = compute_max_drawdown(d['prediction'])

    print('ALL VALID: sharpe ratio = {:.4f}, corr mean = {:.4f}, corr std = {:.4f}, max drawdown = {:.4f}'.format(me / sd, me, sd, max_drawdown))

    

    # first valid eras

    idx = np.where(valid_df["valid2"] == False)[0]

    d = valid_df.iloc[idx].groupby('era')[['target', 'prediction']].corr().iloc[0::2,-1].reset_index()

    me = d['prediction'].mean()

    sd = d['prediction'].std()

    max_drawdown = compute_max_drawdown(d['prediction'])

    print('VALID 1: sharpe ratio = {:.4f}, corr mean = {:.4f}, corr std = {:.4f}, max drawdown = {:.4f}'.format(me / sd, me, sd, max_drawdown))

    

    # second valid eras

    idx = np.where(valid_df["valid2"] == True)[0]

    d = valid_df.iloc[idx].groupby('era')[['target', 'prediction']].corr().iloc[0::2,-1].reset_index()

    me = d['prediction'].mean()

    sd = d['prediction'].std()

    max_drawdown = compute_max_drawdown(d['prediction'])

    print('VALID 2: sharpe ratio = {:.4f}, corr mean = {:.4f}, corr std = {:.4f}, max drawdown = {:.4f}'.format(me / sd, me, sd, max_drawdown))

    

def feature_exposures(valid_df : pd.DataFrame):

    """

    Compute feature exposure

    

    :INPUT:

    - valid_df : pd.DataFrame where at least 2 columns ('prediction' & 'target') exist

    """

    feature_names = [f for f in valid_df.columns

                     if f.startswith("feature")]

    exposures = []

    for f in feature_names:

        fe = spearmanr(valid_df['prediction'], valid_df[f])[0]

        exposures.append(fe)

    return np.array(exposures)



def max_feature_exposure(fe : np.ndarray):

    return np.max(np.abs(fe))



def feature_exposure(fe : np.ndarray):

    return np.sqrt(np.mean(np.square(fe)))



def compute_val_feature_exposure(valid_df : pd.DataFrame):

    """

    Compute feature exposure for valid periods

    

    :INPUT:

    - valid_df : pd.DataFrame where at least 2 columns ('prediction' & 'target') exist

    """

    # all validation

    fe = feature_exposures(valid_df)

    print('ALL VALID: feature exposure = {:.4f}, max feature exposure = {:.4f}'.format(feature_exposure(fe), max_feature_exposure(fe)))

    

    # first valid eras

    idx = np.where(valid_df["valid2"] == False)[0]

    fe = feature_exposures(valid_df.iloc[idx])

    print('VALID 1: feature exposure = {:.4f}, max feature exposure = {:.4f}'.format(feature_exposure(fe), max_feature_exposure(fe)))

    

    # second valid eras

    idx = np.where(valid_df["valid2"] == True)[0]

    fe = feature_exposures(valid_df.iloc[idx])

    print('VALID 2: feature exposure = {:.4f}, max feature exposure = {:.4f}'.format(feature_exposure(fe), max_feature_exposure(fe)))

    

# to neutralize a column in a df by many other columns

def neutralize(df, columns, by, proportion=1.0):

    scores = df.loc[:, columns]

    exposures = df[by].values



    # constant column to make sure the series is completely neutral to exposures

    exposures = np.hstack(

        (exposures,

         np.asarray(np.mean(scores)) * np.ones(len(exposures)).reshape(-1, 1)))



    scores = scores - proportion * exposures.dot(

        np.linalg.pinv(exposures).dot(scores))

    return scores / scores.std()





# to neutralize any series by any other series

def neutralize_series(series, by, proportion=1.0):

    scores = series.values.reshape(-1, 1)

    exposures = by.values.reshape(-1, 1)



    # this line makes series neutral to a constant column so that it's centered and for sure gets corr 0 with exposures

    exposures = np.hstack(

        (exposures,

         np.array([np.mean(series)] * len(exposures)).reshape(-1, 1)))



    correction = proportion * (exposures.dot(

        np.linalg.lstsq(exposures, scores, rcond=None)[0]))

    corrected_scores = scores - correction

    neutralized = pd.Series(corrected_scores.ravel(), index=series.index)

    return neutralized



def unif(df):

    x = (df.rank(method="first") - 0.5) / len(df)

    return pd.Series(x, index=df.index)



def get_feature_neutral_mean(df):

    feature_cols = [c for c in df.columns if c.startswith("feature")]

    df.loc[:, "neutral_sub"] = neutralize(df, [PREDICTION_NAME],

                                          feature_cols)[PREDICTION_NAME]

    scores = df.groupby("era").apply(

        lambda x: np.corrcoef(x["neutral_sub"].rank(pct=True, method="first"), x[TARGET_NAME])).mean()

    return np.mean(scores)



def compute_val_mmc(valid_df : pd.DataFrame):    

    # MMC over validation

    mmc_scores = []

    corr_scores = []

    for _, x in valid_df.groupby("era"):

        series = neutralize_series(pd.Series(unif(x[PREDICTION_NAME])),

                                   pd.Series(unif(x[EXAMPLE_PRED])))

        mmc_scores.append(np.cov(series, x[TARGET_NAME])[0, 1] / (0.29 ** 2))

        corr_scores.append(np.corrcoef(unif(x[PREDICTION_NAME]).rank(pct=True, method="first"), x[TARGET_NAME]))



    val_mmc_mean = np.mean(mmc_scores)

    val_mmc_std = np.std(mmc_scores)

    val_mmc_sharpe = val_mmc_mean / val_mmc_std

    corr_plus_mmcs = [c + m for c, m in zip(corr_scores, mmc_scores)]

    corr_plus_mmc_sharpe = np.mean(corr_plus_mmcs) / np.std(corr_plus_mmcs)

    corr_plus_mmc_mean = np.mean(corr_plus_mmcs)



    print("MMC Mean = {:.6f}, MMC Std = {:.6f}, CORR+MMC Sharpe = {:.4f}".format(val_mmc_mean, val_mmc_std, corr_plus_mmc_sharpe))



    # Check correlation with example predictions

    corr_with_example_preds = np.corrcoef(valid_df[EXAMPLE_PRED].rank(pct=True, method="first"),

                                          valid_df[PREDICTION_NAME].rank(pct=True, method="first"))[0, 1]

    print("Corr with example preds: {:.4f}".format(corr_with_example_preds))
# prediction for valid periods   

pred = model.predict(valid[features])

valid_df = valid4score(valid, pred)

print(valid_df.shape)

valid_df.head()
# compute validation scores

print('Rank correlation -------------------------------')

compute_val_corr(valid_df) # rank correlation

print('Sharpe -------------------------------')

compute_val_sharpe(valid_df) # sharpe

print('Feature exposure -------------------------------')

compute_val_feature_exposure(valid_df) # feature exposure
public_id = "NYANNYAN" # replace with yours

secret_key = "WANWAN" # replace with yours

model_id = "KOKEKOKKOOOO" # replace with yours

PREDICTION_NAME = "prediction_kazutsugi" # 現在はこれ（いずれprediction_nomiになるらしい）

OUTPUT_DIR = '' # prediction dataframeを保存するpath



def submit(tournament : pd.DataFrame, pred : np.ndarray, model_id='abcde'):

    predictions_df = tournament["id"].to_frame()

    predictions_df[PREDICTION_NAME] = pred

    

    # to rank

    predictions_df[PREDICTION_NAME] = predictions_df[PREDICTION_NAME].rank(pct=True, method="first")

    

    # save

    predictions_df.to_csv(pathlib.Path(OUTPUT_DIR + f"predictions_{model_id}.csv"), index=False)

    

    # Upload your predictions using API

    napi = numerapi.NumerAPI(public_id=public_id, secret_key=secret_key)

    submission_id = napi.upload_predictions(pathlib.Path(OUTPUT_DIR + f"predictions_{model_id}.csv"), model_id=model_id)

    print('submitted to {model_id}', model_id=model_id)

    

    return predictions_df
# prediction

pred = model.predict(test[features])

plt.hist(pred);
# submit!（本当に提出する人はコメントアウトしてください）

# predictions_df = submit(tournament, pred, model_id=model_id)