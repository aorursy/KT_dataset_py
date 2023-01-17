import catboost as cb

import numpy as np

import os

import pandas as pd

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from typing import Any, Tuple

import warnings



%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



os.listdir("/kaggle/input/cat-in-the-dat")
DEBUG = True

GBDT = "catboost"
INPUT_PREFIX = "/kaggle/input/cat-in-the-dat"

TRAIN_PATH = os.path.join(INPUT_PREFIX, "train.csv")

TEST_PATH = os.path.join(INPUT_PREFIX, "test.csv")

SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_PREFIX, "sample_submission.csv")



LGB_PARAMS = {}

XGB_PARAMS = {}

if DEBUG is True:

    CB_PARAMS = {

        "loss_function": "Logloss",

        "eval_metric": "AUC",

        "n_estimators": 10,

        "learning_rate": 0.02,

        "random_state": 42,

        "use_best_model": True,

        "depth": 7,

    }

else:

    CB_PARAMS = {

        "loss_function": "Logloss",

        "eval_metric": "AUC",

        "n_estimators": 4000,

        "learning_rate": 0.02,

        "random_state": 42,

        "use_best_model": True,

        "depth": 7,

    }

ES_ROUNDS = 1000

VERBOSE = 500

N_FOLDS = 5

EVAL_RATE = 0.05



CAT_FEATURES = [

    "bin_0",

    "bin_1",

    "bin_2",

    "bin_3",

    "bin_4",

    "nom_0",

    "nom_1",

    "nom_2",

    "nom_3",

    "nom_4",

    "nom_5",

    "nom_6",

    "nom_7",

    "nom_8",

    "nom_9",

    "ord_0",

    "ord_1",

    "ord_2",

    "ord_3",

    "ord_4",

    "ord_5",

    "day",

    "month",

]

TARGET_COL = "target"

REMOVE_COLS = []



pd.set_option("display.max_columns", 25)

warnings.filterwarnings("ignore")
class Loader:

    def __init__(self):

        self



    def _load_csv(self, path: str, nrows: int) -> pd.DataFrame:

        return pd.read_csv(path, nrows=nrows).set_index("id")



    def load_train(self, path: str = TRAIN_PATH, nrows: int = None) -> pd.DataFrame:

        return self._load_csv(path, nrows=nrows)



    def load_test(self, path: str = TEST_PATH, nrows: int = None) -> pd.DataFrame:

        return self._load_csv(path, nrows=nrows)



    def load_submission(

        self, path: str = SAMPLE_SUBMISSION_PATH, nrows: int = None

    ) -> pd.DataFrame:

        return self._load_csv(path, nrows=nrows)
class Preprocesser:

    def __init__(self):

        self



    def run(

        self, train: pd.DataFrame, test: pd.DataFrame

    ) -> Tuple[pd.DataFrame, np.array, pd.DataFrame]:

        X_train, y_train = train.drop(TARGET_COL, axis=1), train[TARGET_COL].values

        X_test = test.copy()

        return (X_train, y_train, X_test)
class Estimator:

    def __init__(self):

        self.lgb_params = LGB_PARAMS

        self.xgb_params = XGB_PARAMS

        self.cb_params = CB_PARAMS

        self.es_rounds = ES_ROUNDS

        self.verbose = VERBOSE

        self.cat_features = CAT_FEATURES

        self.gbdt = GBDT

        self.model = None

        self.models = []

        self.kfolds = []

        self.tr_idxs = []

        self.val_idxs = []

        self.pred_eval = None

        self.pred_valid = None

        self.pred_test = None



    def fit_(

        self,

        X_train: pd.DataFrame,

        y_train: np.array,

        X_eval: pd.DataFrame = None,

        y_eval: np.array = None,

    ) -> None:

        self.model = cb.CatBoostClassifier(**self.cb_params)

        if X_eval is not None:

            self.model.fit(

                X_train,

                y_train,

                eval_set=[(X_train, y_train), (X_eval, y_eval)],

                early_stopping_rounds=self.es_rounds,

                verbose=self.verbose,

                cat_features=self.cat_features,

            )

        else:

            self.model.fit(X_train, y_train, cat_features=self.cat_features)

        self.models.append(self.model)



    def predict_(self, model: Any, X: np.array) -> np.array:

        return model.predict(X, prediction_type="Probability")[:, 1]



    def evaluate_(self, y_true: np.array, y_pred: np.array) -> float:

        return roc_auc_score(y_true, y_pred)



    def kfold_fit(

        self,

        X: pd.DataFrame,

        y: np.array,

        n_splits: int = N_FOLDS,

        shuffle: bool = True,

    ) -> None:

        self.kfolds = StratifiedKFold(

            n_splits=n_splits, random_state=42, shuffle=shuffle

        )

        train_length = len(X)

        eval_length = int(train_length * EVAL_RATE)

        self.pred_eval = np.zeros((eval_length))

        eval_aucs = []

        for fold_idx, (tr_idx, val_idx) in enumerate(self.kfolds.split(X, y)):

            self.tr_idxs.append(tr_idx)

            self.val_idxs.append(val_idx)

            X_tr, y_tr = X.iloc[tr_idx, :], y[tr_idx]

            X_train, X_eval = (

                X_tr.head(train_length - eval_length),

                X_tr.tail(eval_length),

            )

            y_train, y_eval = y_tr[: (train_length - eval_length)], y_tr[-eval_length:]

            del X_tr, y_tr

            self.fit_(X_train, y_train, X_eval, y_eval)

            eval_auc = self.evaluate_(y_eval, self.predict_(self.model, X_eval))

            eval_aucs.append(eval_auc)

            print(f"fold {fold_idx+1} eval_auc: {eval_auc}")

            del X_train, y_train, X_eval, y_eval

        print(f"\nmean eval_auc: {np.mean(eval_aucs)}")



    def kfold_predict(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:

        self.pred_valid = np.zeros((len(X_train)))

        self.pred_test = np.zeros((len(X_test)))

        for fold_idx in range(self.kfolds.n_splits):

            val_idx = self.val_idxs[fold_idx]

            model = self.models[fold_idx]

            valid_pred = self.predict_(model, X_train.iloc[val_idx, :].values)

            test_pred = self.predict_(model, X_test.values)

            self.pred_valid[val_idx] = valid_pred

            self.pred_test += test_pred / self.kfolds.n_splits



    def kfold_feature_importance(self) -> pd.DataFrame:

        df_fi = pd.DataFrame()

        for i, model in enumerate(self.models):

            features = model.feature_names_

            importances = model.feature_importances_

            df_tmp = pd.DataFrame(

                {"feature": features, f"importance_{i}": importances}

            ).set_index("feature")

            if i == 0:

                df_fi = df_tmp.copy()

            else:

                df_fi = df_fi.join(df_tmp, how="left", on="feature")

            del df_tmp

        df_fi["importance"] = df_fi.values.mean(axis=1)

        df_fi.sort_values("importance", ascending=False, inplace=True)

        df_fi.reset_index(inplace=True)

        return df_fi

    

    def plot_feature_importance(self):

        df_fi = self.kfold_feature_importance()

        sns.set()

        plt.figure(figsize=(6,10))

        sns.barplot(y=df_fi["feature"], x=df_fi["importance"])

        plt.tight_layout()

        plt.show()
class Submitter:

    def __init__(self, pred: np.array, filename: str = "submission.csv") -> None:

        loader = Loader()

        self.df = loader.load_submission()

        self.df["target"] = pred

        self.df.to_csv(f"{filename}")

        print("file saved")
%%time

loader = Loader()

preprocesser = Preprocesser()

estimator = Estimator()



train = loader.load_train()

test = loader.load_test()

X_train, y_train, X_test = preprocesser.run(train, test)

del train, test
%%time

estimator.kfold_fit(X_train, y_train)

estimator.kfold_predict(X_train, X_test)

valid_auc = estimator.evaluate_(y_train, estimator.pred_valid)

print(f"\nvalid_auc: {valid_auc}")
estimator.plot_feature_importance()
Submitter(estimator.pred_test, filename="submission.csv").df.head()