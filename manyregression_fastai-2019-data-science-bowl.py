import os

from collections import Counter, defaultdict

from pathlib import Path



from tqdm.notebook import tqdm

import json

import numpy as np

import pandas as pd

from fastai.tabular import * 



pd.set_option('display.max_colwidth', 200)

pd.set_option('display.max_columns', None)

pd.set_option('display.min_rows', 100)

pd.set_option('display.max_rows', 100)

home = Path("/kaggle/input/data-science-bowl-2019/")
specs = pd.read_csv(home/"specs.csv"); len(specs)

specs.head()
train_labels = pd.read_csv(home/"train_labels.csv"); len(train_labels)

train_labels.head(5)
pd.read_csv(home/"sample_submission.csv").head()
%%time

types = {"event_code": np.int16, "event_count": np.int16, "game_time": np.int32}

raw_train = pd.read_csv(home/"train.csv", dtype=types)

raw_train["timestamp"] = pd.to_datetime(raw_train["timestamp"]); len(raw_train)
raw_test = pd.read_csv(home/"test.csv", dtype=types)

raw_test["timestamp"] = pd.to_datetime(raw_test["timestamp"])

raw_test.head(5)
raw_train.sample(5)
# raw_train[raw_train["game_session"] == "969a6c0d56aa4683"].tail()
# Remove `installation_id` without any assesments

ids_with_subms = raw_train[raw_train.type == "Assessment"][['installation_id']].drop_duplicates()

raw_train = pd.merge(raw_train, ids_with_subms, on="installation_id", how="inner"); len(raw_train)
# Reduce event_id to make data preparation faster



specs['hashed_info']=specs['info'].transform(hash)

unique_specs=pd.DataFrame(specs[['hashed_info']].drop_duplicates())

unique_specs["id"] = np.arange(len(unique_specs))

specs = pd.merge(specs,unique_specs,on='hashed_info',how='left')

event_id_mapping = dict(zip(specs.event_id,specs.id))

raw_train["event_id"] = raw_train["event_id"].map(event_id_mapping)

raw_test["event_id"] = raw_test["event_id"].map(event_id_mapping)
def get_accuracy(correct_data):

    # Rounding correct > 1 to 1 lowers the score. Why?

    correct = len(correct_data.loc[correct_data])

    wrong = len(correct_data.loc[~correct_data])

    accuracy = correct/(correct + wrong) if correct + wrong else 0

    return accuracy, correct, wrong



def get_group(accuracy):

    if not accuracy:

        return 0

    elif accuracy == 1:

        return 3

    elif accuracy >= 0.5:

        return 2

    return 1
# I prefer this over calculating average

def lin_comb(v1, v2, beta): return beta*v1 + (1-beta)*v2
def prepare(data: pd.DataFrame, one_hot: List[str], test=False) -> pd.DataFrame:

    one_hot_dict = defaultdict(int)



    prepared = []

    for id_, g in tqdm(data.groupby("installation_id", sort=False)):

        features = process_id(g, one_hot, one_hot_dict.copy(), test)

        if not features:

            continue

        if test:

            features[-1]["is_test"] = 1

        prepared.extend(features)

    return pd.DataFrame(prepared).fillna(0).sort_index(axis=1)
def process_id(id_data: pd.DataFrame, one_hot_cols, one_hot_dict, test: bool) -> pd.DataFrame:

    a_accuracy, a_group, a_correct, a_wrong, counter, accumulated_duration_mean = 0, 0, 0, 0, 0, 0

    a_groups = {"0":0, "1":0, "2":0, "3":0}

    a_durations = defaultdict(int)

    features = []



    for s, gs in id_data.groupby("game_session", sort=False):

        def update_counter(counter: dict, column: str):

            session_counter = Counter(gs.loc[:, column])

            for value in session_counter.keys():

                counter[f"{column}_{value}"] += session_counter[value]

            return counter



        def process_session(gs):

            # share state with parent process_id()

            nonlocal one_hot_dict, a_groups, a_durations, a_accuracy, a_group, a_correct, a_wrong, counter, accumulated_duration_mean

            # increment one hot columns for session, e.g. Bird Measurer: 50

            def accumulate():

                nonlocal accumulated_duration_mean

                # accumulated one_hot features per id for a given session, e.g. Bird Measurer: 50

                for c in one_hot_cols:

                    one_hot_dict.update(update_counter(one_hot_dict, c))

                duration = (gs["timestamp"].iloc[-1] - gs["timestamp"].iloc[0]).seconds

                # an accumulated session duration mean

                accumulated_duration_mean = lin_comb(accumulated_duration_mean or duration,

                                                     duration, beta=0.9)

                a_durations[f"duration_{gs.title.iloc[0]}"] = duration

                

            if gs["type"].iloc[0] != "Assessment":

                accumulate()

                return



            guess_mask = ((gs["event_data"].str.contains("correct")) & 

             (((gs["event_code"] == 4100) &(~gs["title"].str.startswith("Bird")) | 

               ((gs["event_code"] == 4110) & (gs["title"].str.startswith("Bird"))))))

            answers = gs.loc[guess_mask, "event_data"].apply(lambda x: json.loads(x).get("correct"))



            # skip assessments without attempts in train

            if answers.empty and not test:

                accumulate()

                return



            accuracy, correct, wrong = get_accuracy(answers)

            group = get_group(accuracy)

            processed = {"installation_id": id_data["installation_id"].iloc[0],

                         "title": gs["title"].iloc[0],

                         "timestamp": gs["timestamp"].iloc[0],

                         "accumulated_duration_mean": accumulated_duration_mean,

                         "accumulated_correct": a_correct, "accumulated_incorrect": a_wrong,

                         "accumulated_accuracy_mean": a_accuracy/counter if counter > 0 else 0,

                         "accumulated_accuracy_group_mean": a_group/counter if counter > 0 else 0, 

                         "accuracy_group": group,

                        }

            processed.update(a_groups)

            processed.update(one_hot_dict)

            processed.update(a_durations)

            counter += 1

            a_accuracy += accuracy

            a_correct += correct

            a_wrong += wrong

            a_group += group

            a_groups[str(group)] += 1

            accumulate()

            return processed

        

        # skip sessions with 1 row

        if len(gs) == 1 and not test:

            continue

        gs.reset_index(inplace=True, drop=True)

        if (gs["timestamp"].iloc[-1] - gs["timestamp"].iloc[0]).seconds > 1800:

            gs["passed"] = gs.loc[:, "timestamp"].diff().apply(lambda x: x.seconds)

            id_max = gs["passed"].idxmax()

            if gs["passed"].max() > 1800:

                session = gs.iloc[:id_max]

                continued_session = gs.iloc[id_max:]

                fs = process_session(session)

                c_fs = process_session(continued_session)

                if fs:

                    features.append(fs)

                if c_fs:

                    features.append(c_fs)

                continue



        session_features = process_session(gs)

        if session_features:

            features.append(session_features)

        

    return features
one_hot_counters=["title", "type", "event_code", "event_id"]

train = prepare(raw_train, one_hot_counters)

# train = prepare(raw_train.iloc[:1_000_000], one_hot_counters)
add_datepart(train, "timestamp", prefix="timestamp_", time=True)

train.head()
test = prepare(raw_test, one_hot=one_hot_counters, test=True)
# for the case when one hot encoded columns don't match between datasets

add_datepart(test, "timestamp", prefix="timestamp_", time=True);
# why discard good data from test, let's use all the taken assessments in train!

train = (pd.concat([train, test[test["is_test"] == 0].drop(columns=["is_test"])],

                   ignore_index=True, sort=False)).fillna(0)

train.head()
test = test.loc[test["is_test"] == 1].reset_index(drop=True)

test.drop(columns=["accuracy_group", "is_test"], inplace=True)

test.head()
diff = train.drop(columns=["accuracy_group"]).columns.difference(test.columns)

display(f"Test doesn't contain {diff.values}")

display(f"Train doesn't contain {test.columns.difference(train.columns).values}")

train.drop(columns=diff, inplace=True)
main_train = train.copy()

# train = main_train.copy()
del_cols = ["timestamp_Second"]

for col in train.columns.values:

    counts = train[col].value_counts().iloc[0]

    if (counts / train.shape[0]) >= 0.99:

        del_cols.append(col)

train.drop(columns=del_cols, inplace=True, errors="ignore")

test.drop(columns=del_cols, inplace=True, errors="ignore")

display(f"Dropped {del_cols}")
procs = [FillMissing, Categorify, Normalize]
np.random.seed(42)
# remove outliers

train = train[train[train.columns[train.columns.str.startswith("duration_", na=False)].to_list()].apply(sum, axis=1) < 10000].reset_index(drop=True)
# grab the last assessments per id

valid_idx = [g.iloc[-1].name for i, g in train.groupby("installation_id", sort=False)]; len(valid_idx)
train.accuracy_group.value_counts(normalize=True)
train.loc[valid_idx].accuracy_group.value_counts(normalize=True)
train.title.value_counts(normalize=True)
train.loc[valid_idx].title.value_counts(normalize=True)
date_cols = train.columns[train.columns.str.startswith("timestamp_", na=False)].to_list()
dep_var = "accuracy_group"

cat_names = list(filter(lambda x: x not in ["timestamp_Elapsed"], date_cols)) + ["title"]

cont_names = list(filter(lambda x: x not in ["installation_id", dep_var] + cat_names,

                         train.columns.to_list()))
data = (TabularList.from_df(train, path="/kaggle/working", cat_names=cat_names, cont_names=cont_names, procs=procs)

        .split_by_idx(valid_idx=valid_idx)

        .label_from_df(cols=dep_var, label_cls=FloatList)

        .add_test(TabularList.from_df(test, path=home, cat_names=cat_names, cont_names=cont_names, procs=procs))

        .databunch()

)
# data.show_batch()
from functools import partial

import scipy as sp

from sklearn.metrics import cohen_kappa_score



class OptimizedRounder():

    """

    An optimizer for rounding thresholds

    to maximize Quadratic Weighted Kappa (QWK) score

    # https://www.kaggle.com/naveenasaithambi/optimizedrounder-improved

    """

    def __init__(self, initial_coef, labels):

        self.coef_ = 0

        self.initial_coef = initial_coef

        self.labels = labels



    def _kappa_loss(self, coef, X, y):

        """

        Get loss according to

        using current coefficients

        

        :param coef: A list of coefficients that will be used for rounding

        :param X: The raw predictions

        :param y: The ground truth labels

        """

        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = self.labels)

        return -cohen_kappa_score(X_p, y, weights="quadratic")



    def fit(self, X, y):

        """

        Optimize rounding thresholds

        

        :param X: The raw predictions

        :param y: The ground truth labels

        """

        loss_partial = partial(self._kappa_loss, X=X, y=y)

        self.coef_ = sp.optimize.minimize(loss_partial, self.initial_coef, method='nelder-mead')



    def predict(self, X, coef):

        """

        Make predictions with specified thresholds

        

        :param X: The raw predictions

        :param coef: A list of coefficients that will be used for rounding

        """

        return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = self.labels)



    def coefficients(self): return self.coef_['x']
from fastai.metrics import RegMetrics



class KappaScoreRegression(RegMetrics):

    def on_epoch_end(self, last_metrics, **kwargs):

        preds = self.preds.flatten()

        opt = OptimizedRounder([0.5, 1.5, 2.0], labels=[0, 1, 2, 3])

        opt.fit(preds, self.targs)

        coefs = opt.coefficients()

        def rounder(preds):

            y = preds.clone()

            y[y < coefs[0]] = 0

            y[y >= coefs[2]] = 3

            y[(y >= coefs[0]) & (y < coefs[1])] = 1

            y[(y >= coefs[1]) & (y < coefs[2])] = 2

            return y.type(torch.IntTensor)



        qwk = cohen_kappa_score(rounder(preds), self.targs, weights="quadratic")

        return add_metrics(last_metrics, qwk)
from fastai.callbacks import *



learn = tabular_learner(data, layers=[2000,100],

                        metrics=[KappaScoreRegression()],

                        y_range=[0, 3],

                        emb_drop=0.04,

                        ps=0.6,

                        callback_fns=[partial(EarlyStoppingCallback, monitor="kappa_score_regression", mode="max", patience=7),

                                      partial(SaveModelCallback, monitor="kappa_score_regression", mode="max", name="best_model")]

                       )
# learn.lr_find()

# learn.recorder.plot()
learn.fit_one_cycle(30, 3e-03)
preds_train, y = learn.get_preds(ds_type=DatasetType.Valid)

labels_train = preds_train.flatten()

opt = OptimizedRounder([0.5, 1.5, 2.0], labels=[0, 1, 2, 3])

opt.fit(labels_train, y)
coefs = opt.coefficients(); coefs
def rounder(preds):

    y = preds.clone()

    y[y < coefs[0]] = 0

    y[y >= coefs[2]] = 3

    y[(y >= coefs[0]) & (y < coefs[1])] = 1

    y[(y >= coefs[1]) & (y < coefs[2])] = 2

    return y.type(torch.IntTensor)
preds, y = learn.get_preds(ds_type=DatasetType.Test)

labels = preds.flatten()
labels = rounder(labels)
submission = pd.DataFrame({"installation_id": test.installation_id, "accuracy_group": labels})

submission.to_csv("submission.csv", index=False)

len(submission), submission.accuracy_group.value_counts()