import gc

import os

from pathlib import Path

import sys

import collections



import pandas as pd

import numpy as np

import scipy as sp



import matplotlib.pyplot as plt



from tqdm import tqdm_notebook as tqdm



import joblib



from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline

# from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder, QuantileTransformer #, MinMaxScaler

# from sklearn.experimental import enable_iterative_imputer

# from sklearn.impute import SimpleImputer #, IterativeImputer, MissingIndicator

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_selection import SelectFromModel

from sklearn.decomposition import TruncatedSVD

from sklearn.metrics import mean_squared_error



import category_encoders as ce

import lightgbm as lgbm

%matplotlib inline

plt.rcParams["figure.figsize"] = (15, 5)

pd.options.display.max_columns = 50
VERSION = "1.6.6"
# Is this environment Kaggle kernel?

IS_KAGGLE = "KAGGLE_URL_BASE" in os.environ

print(f"IS_KAGGLE: {IS_KAGGLE}")



# GPU enviroment?

USE_GPU = "NVIDIA_VISIBLE_DEVICES" in os.environ

print(f"USE_GPU: {USE_GPU}")



# Save and load interim data?

USE_CACHE = False
model_dir = Path("." if IS_KAGGLE else "../output/models")

model_dir.mkdir(parents=True, exist_ok=True)
def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / (1024 ** 2)    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

#                 if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

#                     df[col] = df[col].astype(np.float16)

                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:

        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(

            end_mem, 100 * (start_mem - end_mem) / start_mem)

        )

        

    return df
train = pd.read_csv("../input/exam-for-students20200129/train.csv", index_col=0, na_values="").pipe(reduce_mem_usage)

X_test = pd.read_csv("../input/exam-for-students20200129/test.csv", index_col=0, na_values="").pipe(reduce_mem_usage)
# 小数点がほとんどコンマなので、decimalをコンマ指定して読み込み

df_country = pd.read_csv("../input/exam-for-students20200129/country_info.csv", decimal=",").pipe(reduce_mem_usage)
# GDPだけ小数点がdotで数値として読めていないので、数値に変換

df_country["GDP ($ per capita)"] = df_country["GDP ($ per capita)"].astype(np.float32)
df_country
df_country.info()
train
train = train.reset_index().merge(df_country, how="left", on="Country").set_index("Respondent")

X_test = X_test.reset_index().merge(df_country, how="left", on="Country").set_index("Respondent")
train.info()
X_train = train.drop(columns="ConvertedSalary")

# log変換してRMSLEで評価しやすくする

y_train = np.log1p(train.ConvertedSalary)
split_cols = [

  "DevType",

  "CommunicationTools",

  "FrameworkWorkedWith",

  "AdsActions",

  "ErgonomicDevices",

  "Gender",

  "SexualOrientation",

  "RaceEthnicity"

]
def flatten(l):

    for el in l:

        if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):

            yield from flatten(el)

        else:

            yield el
def split_text(df):

    for col in split_cols:

        categories = list(set(flatten(X_train[col].str.split(";").tolist())))

        categories = [i for i in categories if str(i) != 'nan']

        for category in categories:

            df[f"{col}_{category}"] = df[col].str.contains(category).astype(np.float32)

        # count ;

        df[f"count_{col}"] = df[col].str.count(";")

    return df
X_train = split_text(X_train)

X_test = split_text(X_test)
X_train.info()
# LGBMでエラーになるのでリネームしておく

X_train.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in X_train.columns]

X_test.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in X_test.columns]
object_cols = X_train.select_dtypes(include="object").columns.tolist()
class BaseTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):

        pass



    def fit(self, X, y=None):

        return self



    def transform(self, X):

        return self



    def get_feature_names(self):

        pass



class KFoldTargetEncoder(BaseTransformer):

    """K-Fold target encoder"""



    def __init__(self, cols=None,

                 n_splits=5, random_state=24, shuffle=True, **kwargs):

        super().__init__()

        if cols is None:

            cols = []

        self.cols = cols

        self.enc = ce.TargetEncoder(cols=cols, return_df=False, **kwargs)

        self.n_splits = n_splits

        self.random_state = random_state

        self.shuffle = shuffle

        self.kwargs = kwargs

        self.train_idx = None

        self.y = None



    def fit(self, X, y=None):

        self.enc.fit(X[self.cols], y)

        self.train_idx = X.index

        self.y = y

        return self



    def transform(self, X):

        if self.train_idx is X.index:

            # training data の変換

            df = pd.DataFrame(index=X.index, columns=self.cols, dtype=float)

            skf = KFold(n_splits=self.n_splits, random_state=self.random_state, shuffle=self.shuffle)

            for train_idx, val_idx in skf.split(X, self.y):

                enc = ce.TargetEncoder(cols=self.cols, return_df=False, **self.kwargs)

                X_train, y_train = X.loc[X.index[train_idx], self.cols], self.y.iloc[train_idx]

                X_val = X.loc[X.index[val_idx], self.cols]

                enc.fit(X_train, y_train)

                df.loc[df.index[val_idx], self.cols] = enc.transform(X_val)

            return df

        # test data の変換

        return self.enc.transform(X[self.cols])



    def get_feature_names(self):

        return self.cols

feature_union1 = FeatureUnion([

    ("te", KFoldTargetEncoder(

        cols=object_cols,

        smoothing=.8

    )),

], n_jobs=None, verbose=True)
X_train1 = feature_union1.fit_transform(X_train, y_train)

X_test1 = feature_union1.transform(X_test)
oe = ce.OrdinalEncoder(cols=object_cols)
X_train = oe.fit_transform(X_train, y_train)

X_test = oe.transform(X_test)
feature_names = X_train.columns.tolist()
X_train = np.hstack([X_train, X_train1])

X_test = np.hstack([X_test, X_test1])
feature_names += feature_union1.get_feature_names()
X_train.shape
# Kfold lgbm



models = []



seeds = [114]



for seed in seeds:

    params = {

        "objective": "regression",

        "learning_rate": .02,

        "tree_learner": "data",

        "device_type": "cpu",

        "num_leaves": 128,

        "seed": seed,

        "colsample_bytree": .8,

        "max_depth": 7,

        "subsample": .9,

    #     "min_data_in_leaf": 128,

        "metric": ["rmse"]

    }



    skf = KFold(n_splits=5, random_state=seed, shuffle=True)



    for i, (train_ix, test_ix) in enumerate(skf.split(X_train, y_train)):

        X_train_, y_train_ = X_train[train_ix,], y_train.values[train_ix]

        X_val_, y_val_ = X_train[test_ix,], y_train.values[test_ix]



        train_data = lgbm.Dataset(

            data=X_train_,

            label=y_train_,

            feature_name=feature_names

        )



        val_data = lgbm.Dataset(

            data=X_val_,

            label=y_val_,

            reference=train_data,

            feature_name=feature_names

        )



        model = lgbm.train(params=params, train_set=train_data, num_boost_round=9999, valid_sets=[val_data],

                           early_stopping_rounds=200, verbose_eval=100)

        models.append(model)

        y_pred_ = model.predict(X_val_)

        score = mean_squared_error(y_val_, y_pred_)



        print('CV Score of Fold_%d is %f' % (i, score))
joblib.dump(models, model_dir / f"models-lgbm-{VERSION}.joblib")
for i, model in enumerate(models):

    if i == 0:

        y_preds = model.predict(X_test)

    else:

        y_preds = np.vstack((y_preds, model.predict(X_test)))
# targetはlog変換されているので, expで元に戻す

y_preds = np.expm1(y_preds)
y_pred = y_preds.mean(axis=0)
submission = pd.read_csv('../input/exam-for-students20200129/sample_submission.csv', index_col=0)

submission.ConvertedSalary = y_pred

submission.to_csv(model_dir / f'submission-{VERSION}.csv')
submission
for i, model in enumerate(models):

    if i == 0:

        feature_importances = pd.DataFrame({f"m{i}": pd.Series(model.feature_importance(importance_type="gain"),

                                                               index=model.feature_name())},

                                          index=model.feature_name())

    else:

        feature_importances[f"m{i}"] = pd.Series(model.feature_importance(importance_type="gain"),

                                                 index=model.feature_name())
feature_importances["mean"] = feature_importances.mean(axis=1)
with pd.option_context("display.max_rows", 1000):

    print(feature_importances["mean"].sort_values(ascending=False))