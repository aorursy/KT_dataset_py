# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import copy

import numpy as np

import pandas as pd

import warnings

from sklearn.model_selection import StratifiedKFold

from sklearn import metrics

import lightgbm as lgb_r

from lightgbm import Dataset

from optuna.integration import lightgbm as lgb

from catboost import CatBoostClassifier, Pool

from sklearn.linear_model import LinearRegression

import tensorflow as tf



np.random.seed(42)

warnings.filterwarnings("ignore")
TRAIN_RAW_PATH = "../input/ai-medical-contest-2020/train.csv"

TEST_RAW_PATH = "../input/ai-medical-contest-2020/test.csv"

SUB_RAW_PATH = "../input/ai-medical-contest-2020/sample_submission.csv"

COLS_CAT = [  # カテゴリ変数のリスト

    "type_hospital",

    "place_hospital",

    "place_patient_birth",

    "place_patient_live",

    "place_patient_live2",

    "patient_type",

    "icu",

    "intubed",

    "sex",

    "asthma",

    "cardiovascular",

    "chronic_renal_failure",

    "copd",

    "diabetes",

    "hypertension",

    "immunosuppression",

    "obesity",

    "pneumonia",

    "pregnancy",

    "other_disease",

    "tobacco",

    "contact_other_covid",

    "test_result",

]
class Preprocessing:

    def __init__(self, train_path, test_path, include_cat=True):

        self.df_train = pd.read_csv(train_path)

        self.df_test = pd.read_csv(test_path)

        self.df_traintest = self.make_traintest()

        self.df_traintest = self.preprocessing(include_cat)

        self.df_traintest = self.generate_features()

        self.df_train, self.df_test = self.create_folds()



    def make_traintest(self):

        # traintestファイルを作る

        df_traintest = pd.concat([self.df_train, self.df_test]).reset_index(drop=True)

        format = "%Y-%m-%d"  # 二次地表示のフォーマット, 例) 2020-09-26

        cols_time = ["entry_date", "date_symptoms", "date_died"]  # 日時を表す列名のリスト

        for col in cols_time:  # 各列について

            df_traintest[col] = pd.to_datetime(

                df_traintest[col], format=format

            )  # string型からdatetime型に変換

        return df_traintest



    def preprocessing(self, include_cat=True):

        df_traintest = self.df_traintest

        cols_tmp = [

                "icu",

                "intubed",

                "asthma",

                "cardiovascular",

                "chronic_renal_failure",

                "copd",

                "diabetes",

                "hypertension",

                "immunosuppression",

                "obesity",

                "pneumonia",

                "pregnancy",

                "other_disease",

                "tobacco",

                "contact_other_covid",

            ]

        for col in cols_tmp:

            df_traintest[col] = df_traintest[col].replace("No", 0)

            df_traintest[col] = df_traintest[col].replace("Yes", 1)

            df_traintest[col] = df_traintest[col].fillna(2)

            df_traintest[col] = df_traintest[col].astype(int)

        # intにする

        df_traintest["place_patient_birth"] = df_traintest[

            "place_patient_birth"

        ].fillna(

            33

        )  # nanを新規のカテゴリ 33 に置換

        df_traintest["place_patient_birth"] = df_traintest[

            "place_patient_birth"

        ].astype(int)



        # 日時変数を1月1日から数えた日数に変換する

        df_traintest["entry_date"] = (

            df_traintest["entry_date"].apply(lambda x: x.dayofyear).astype(np.uint16)

        )

        df_traintest["date_symptoms"] = (

            df_traintest["date_symptoms"].apply(lambda x: x.dayofyear).astype(np.uint16)

        )

        

        if include_cat:

            # カテゴリ変数をラベルエンコーディングする (数値に置き換える).

            df_traintest["sex"] = df_traintest["sex"].replace(

                "female", 0

            )  # femaleに0を代入

            df_traintest["sex"] = df_traintest["sex"].replace("male", 1)  # maleに1を代入

            df_traintest["sex"] = df_traintest["sex"].astype(int)  # 型を整数に変換



            df_traintest["patient_type"] = df_traintest["patient_type"].replace(

                "inpatient", 0

            )

            df_traintest["patient_type"] = df_traintest["patient_type"].replace(

                "outpatient", 1

            )

            df_traintest["patient_type"] = df_traintest["patient_type"].astype(int)



            

            df_traintest["test_result"] = df_traintest["test_result"].replace(

                "Negative", 0

            )

            df_traintest["test_result"] = df_traintest["test_result"].replace(

                "Positive", 1

            )

            df_traintest["test_result"] = df_traintest["test_result"].replace(

                "Results awaited", 1

            )

            df_traintest["test_result"] = df_traintest["test_result"].astype(int)

        

        return df_traintest



    def generate_features(self):

        df_traintest = self.df_traintest

        col_index = "patient_id"  # idの列

        # 変数と変数の差をとる

        df_traintest["entry_-_symptom_date"] = (

            df_traintest["entry_date"] - df_traintest["date_symptoms"]

        )  # 発症から入院までの日数



        # 変数と変数の乗算をとる

        age = df_traintest["age"].values

        age = (age - age.mean()) / age.std()  # 年齢を正規化

        entry_date = df_traintest["entry_date"].values

        entry_date = (entry_date - entry_date.mean()) / entry_date.std()  # 年齢を正規化

        df_traintest["age_x_entry_date"] = (

            age * entry_date

        )  # 年齢と入院日を乗算. 2つの変数の相互作用を表現できる



        # カウントエンコーディング

        # あるカテゴリがデータに何件あるか、を特徴量とする. 例) 入院日が2020/1/1の行が何件あるか

        col_groupby = "entry_date"  # カウントを行う列

        df_tmp = copy.deepcopy(df_traintest)

        df_agg = (

            df_traintest.groupby(col_groupby)[col_index].agg(len).reset_index()

        )  # 集約特徴量を得る

        col_new = "entry_date_count"  # 特徴量名. 各日の入院患者数

        df_agg.columns = [col_groupby, col_new]

        df_tmp = pd.merge(df_tmp, df_agg, on=col_groupby, how="left").drop(

            col_groupby, axis=1

        )

        df_traintest[col_new] = df_tmp[col_new]

        return df_traintest



    def create_folds(self):

        df_traintest = self.df_traintest

        df_train = df_traintest.iloc[: len(self.df_train)]

        df_test = df_traintest.iloc[len(self.df_train) :].reset_index(drop=True)

        df_train["skfold"] = -1

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        folds = skf.split(

            np.arange(len(df_train)), y=df_train["died"]

        )  # 各foldターゲットのラベルの分布がそろうようにする = stratified K fold

        for fold, (trn_, val_) in enumerate(folds):

            df_train.loc[val_, "skfold"] = fold



        return (df_train, df_test)
def load_data(df_train, df_test, fold):

    cols_feature = df_train.columns.values.tolist()[3:-1]

    tr_X = df_train[df_train.skfold != fold][cols_feature]

    val_X = df_train[df_train.skfold == fold][cols_feature]

    tr_y = df_train[df_train.skfold != fold]["died"]

    val_y = df_train[df_train.skfold == fold]["died"]

    test_X = df_test[cols_feature]

    return (tr_X, tr_y, val_X, val_y, test_X)
def run_baseline(tr_X, tr_y, val_X, val_y, fold, COLS_CAT):

    params = {

        "objective": "binary",  # 目的->2値分類

        "num_threads": -1,

        "bagging_seed": 42,  # random seed の固定

        "random_state": 42,  # random seed の固定

        "boosting": "gbdt",

        "metric": "auc",  # 評価変数->AUC

        "verbosity": -1,

    }

    train_data = Dataset(

        tr_X, label=tr_y, categorical_feature=COLS_CAT

    )  # LightGBM用にデータを整形

    valid_data = Dataset(val_X, label=val_y, categorical_feature=COLS_CAT)

    model = lgb_r.train(

        params,  # モデルのパラメータ

        train_data,  # 学習データ

        150000,  # 学習を繰り返す最大epoch数, epoch = モデルの学習回数

        valid_sets=[train_data, valid_data],  # 検証データ

        verbose_eval=100,  # 100 epoch ごとに経過を表示する

        early_stopping_rounds=150,  # 150epoch続けて検証データのロスが減らなかったら学習を中断する

    )

    return model
def lgb_optuna(tr_X, tr_y, val_X, val_y, fold, COLS_CAT):

    dtrain = Dataset(

        tr_X, tr_y, categorical_feature=COLS_CAT, free_raw_data=False

    )

    dval = Dataset(

        val_X, val_y, categorical_feature=COLS_CAT, free_raw_data=False

    )

    params = {

        "objective": "binary",  # 目的->2値分類

        "num_threads": -1,

        "bagging_seed": 42,  # random seed の固定

        "random_state": 42,  # random seed の固定

        "boosting": "gbdt",

        "metric": "auc",  # 評価変数->AUC

        "verbosity": -1,

    }



    model = lgb.train(

        params,  # モデルのパラメータ

        dtrain,  # 学習データ

        150000,  # 学習を繰り返す最大epoch数, epoch = モデルの学習回数

        valid_sets=[dtrain, dval],  # 検証データ

        verbose_eval=100,  # 100 epoch ごとに経過を表示する

        early_stopping_rounds=150,  # 150epoch続けて検証データのロスが減らなかったら学習を中断する

    )



    best_params = model.params

    print("Best params:", best_params)

    print("  Params: ")

    for key, value in best_params.items():

        print("    {}: {}".format(key, value))

    return model
def run_cat(train_pool, validate_pool, fold):



    # 学習

    model = CatBoostClassifier(iterations=50, custom_loss=["AUC"])

    model.fit(

        train_pool,

        eval_set=validate_pool,  # 検証用データ

        early_stopping_rounds=150,  # 150回以上精度が改善しなければ中止

        use_best_model=True,

        verbose=False,

    )

    print(model.get_best_score())

    # 予測

    preds = model.predict_proba(validate_pool)[:, 1]

    return model
pp_lgb = Preprocessing(

            TRAIN_RAW_PATH, TEST_RAW_PATH, include_cat=True

        )

df_train = pp_lgb.df_train

df_test = pp_lgb.df_test



preds_val_base = []

preds_test_base = []



for fold in range(0, 5):

    tr_X, tr_y, val_X, val_y, test_X = load_data(df_train, df_test, fold)

    model = run_baseline(tr_X, tr_y, val_X, val_y, fold, COLS_CAT)

    pred_val = model.predict(

        val_X, num_iteration=model.best_iteration

    )  # 検証データに対する予測を実行

    preds_val_base.append(pred_val)

    pred_test = model.predict(

        test_X, num_iteration=model.best_iteration

    )  # テストデータに対する予測を実行

    preds_test_base.append(pred_test)

    score = metrics.roc_auc_score(val_y, pred_val)

    print(score)

preds_test_mean_base = np.array(preds_test_base).mean(

            axis=0

)  # モデルを5個作ったので予測は一つのデータに5個ある. これを平均する.
pp_cat = Preprocessing(

    TRAIN_RAW_PATH, TEST_RAW_PATH, include_cat=False

)

df_train = pp_cat.df_train

df_test = pp_cat.df_test



preds_test_cat = []

preds_val_cat = []



for fold in range(0, 5):

    tr_X, tr_y, val_X, val_y, test_X = load_data(df_train, df_test, fold)

    cat_features = [tr_X.columns.get_loc(col) for col in COLS_CAT]

    train_pool = Pool(tr_X, tr_y, cat_features=cat_features)

    validate_pool = Pool(val_X, val_y, cat_features=cat_features)

    test_pool = Pool(test_X, cat_features=cat_features)

    model = run_cat(train_pool, validate_pool, fold)

    pred_val = model.predict_proba(validate_pool)[:, 1]  # 検証データに対する予測を実行

    preds_val_cat.append(pred_val)

    pred_test = model.predict_proba(test_pool)[:, 1]  # テストデータに対する予測を実行

    preds_test_cat.append(pred_test)

    score = metrics.roc_auc_score(val_y, pred_val)

    print(score)

    

preds_test_mean_cat = np.array(preds_test_cat).mean(

            axis=0

)  # モデルを5個作ったので予測は一つのデータに5個ある. これを平均する.
df_train = pp_lgb.df_train

df_test = pp_lgb.df_test



preds_test_lgb = []

preds_val_lgb = []

for fold in range(0, 5):

    tr_X, tr_y, val_X, val_y, test_X = load_data(df_train, df_test, fold)

    model = lgb_optuna(tr_X, tr_y, val_X, val_y, fold, COLS_CAT)

    pred_val = model.predict(

        val_X, num_iteration=model.best_iteration

    )  # 検証データに対する予測を実行

    preds_val_lgb.append(pred_val)

    pred_test = model.predict(

        test_X, num_iteration=model.best_iteration

    )  # テストデータに対する予測を実行

    preds_test_lgb.append(pred_test)

    score = metrics.roc_auc_score(val_y, pred_val)

    print(score)

preds_test_mean_lgb = np.array(preds_test_lgb).mean(

            axis=0

)  # モデルを5個作ったので予測は一つのデータに5個ある. これを平均する.
def generate_stack_train(preds_val_base, preds_val_lgb, preds_val_cat):

    df_ = df_train[["patient_id", "skfold"]]

    

    id_ = [

        df_[df_["skfold"] == fold]["patient_id"] for fold in range(5)

    ]

    pred_folds = [

        pd.DataFrame(

            {"patient_id": id_[fold], "pred_base": preds_val_base[fold],

            "pred_lgb": preds_val_lgb[fold], "pred_cat": preds_val_cat[fold]}

        )

        for fold in range(5)

    ]

    features_by_model = pd.concat(pred_folds)

    return df_.drop(["skfold"],axis=1).merge(features_by_model, on="patient_id")
def generate_stack_test(preds_test_mean_base, preds_test_mean_lgb, preds_test_mean_cat):

    id_ = df_test["patient_id"]

    

    pred_ = pd.DataFrame(

            {"patient_id": id_, "pred_base": preds_test_mean_base,

            "pred_lgb": preds_test_mean_lgb, "pred_cat": preds_test_mean_cat}

    )

    return pred_
train_stack = generate_stack_train(preds_val_base, preds_val_lgb, preds_val_cat)

test_stack = generate_stack_test(preds_test_mean_base, preds_test_mean_lgb, preds_test_mean_cat)
# NN

model = tf.keras.Sequential([

    tf.keras.layers.Dense(units=1, input_shape=(None,3)),

    tf.keras.layers.Dense(128),

    tf.keras.layers.Dense(64),

    tf.keras.layers.Dense(1, activation='sigmoid')

])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC()])



X = train_stack[["pred_base","pred_cat","pred_lgb"]]

y = df_train["died"]





model.fit(X, y, batch_size=50, epochs=10)



pred = model.predict_proba(X)

score = metrics.roc_auc_score(y, pred)

print(score)

submission_id = pd.read_csv(SUB_RAW_PATH)["patient_id"]

final_pred = np.ravel(model.predict_proba(test_stack[["pred_base","pred_cat","pred_lgb"]]))

pd.DataFrame({'patient_id': submission_id, 'died': final_pred}).to_csv("submission.csv", index=False)
# all

X = train_stack.drop(["patient_id"],axis=1)

y = df_train["died"]

reg = LinearRegression().fit(X,y)

pred = reg.predict(X)

score = metrics.roc_auc_score(y, pred)

print(score)

submission_id = pd.read_csv(SUB_RAW_PATH)["patient_id"]

final_pred = reg.predict(test_stack.drop(["patient_id"], axis=1))

pd.DataFrame({'patient_id': submission_id, 'died': final_pred}).to_csv("submission_all.csv", index=False)
# baseline + lgb_optuna

X = train_stack.drop(["patient_id","pred_cat"],axis=1)

y = df_train["died"]

reg = LinearRegression().fit(X,y)

pred = reg.predict(X)

score = metrics.roc_auc_score(y, pred)

print(score)

submission_id = pd.read_csv(SUB_RAW_PATH)["patient_id"]

final_pred = reg.predict(test_stack.drop(["patient_id","pred_cat"], axis=1))

pd.DataFrame({'patient_id': submission_id, 'died': final_pred}).to_csv("submission_bl.csv", index=False)
# baseline + cat

X = train_stack.drop(["patient_id","pred_lgb"],axis=1)

y = df_train["died"]

reg = LinearRegression().fit(X,y)

pred = reg.predict(X)

score = metrics.roc_auc_score(y, pred)

print(score)

submission_id = pd.read_csv(SUB_RAW_PATH)["patient_id"]

final_pred = reg.predict(test_stack.drop(["patient_id","pred_lgb"], axis=1))

pd.DataFrame({'patient_id': submission_id, 'died': final_pred}).to_csv("submission_bc.csv", index=False)
# lgb_optuna + cat

X = train_stack.drop(["patient_id","pred_base"],axis=1)

y = df_train["died"]

reg = LinearRegression().fit(X,y)

pred = reg.predict(X)

score = metrics.roc_auc_score(y, pred)

print(score)

submission_id = pd.read_csv(SUB_RAW_PATH)["patient_id"]

final_pred = reg.predict(test_stack.drop(["patient_id","pred_base"], axis=1))

pd.DataFrame({'patient_id': submission_id, 'died': final_pred}).to_csv("submission_lc.csv", index=False)
# baseline

submission_id = pd.read_csv(SUB_RAW_PATH)["patient_id"]

final_pred = preds_test_mean_base

pd.DataFrame({'patient_id': submission_id, 'died': final_pred}).to_csv("submission_b.csv", index=False)
# lgb_optuna

submission_id = pd.read_csv(SUB_RAW_PATH)["patient_id"]

final_pred = preds_test_mean_lgb

pd.DataFrame({'patient_id': submission_id, 'died': final_pred}).to_csv("submission_l.csv", index=False)
# cat

submission_id = pd.read_csv(SUB_RAW_PATH)["patient_id"]

final_pred = preds_test_mean_cat

pd.DataFrame({'patient_id': submission_id, 'died': final_pred}).to_csv("submission_c.csv", index=False)