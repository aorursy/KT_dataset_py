import pandas as pd

from pathlib import Path

from sklearn.model_selection import KFold, train_test_split

from lightgbm import LGBMClassifier

import gc
#LightGBM GPU installation didn't work due to "OpenCL device not found :(". Anyone successful with the current setup in Kaggle?

#%%bash

#apt install libboost-all-dev -y

#pip uninstall lightgbm -y

#pip install lightgbm --install-option="--gpu" --install-option="--opencl-include-dir=/usr/local/cuda/include/" --install-option="--opencl-library=/usr/local/cuda/lib64/libOpenCL.so"
data_dir=Path("/kaggle/input/ieee-fraud-detection")



dd=pd.read_csv(data_dir / "train_transaction.csv")

ddid=pd.read_csv(data_dir / "train_identity.csv")

dd=dd.merge(ddid, on="TransactionID", how="left")



ddtest=pd.read_csv(data_dir / "test_transaction.csv")

ddtestid=pd.read_csv(data_dir / "train_identity.csv")

ddtest=ddtest.merge(ddtestid, on="TransactionID", how="left")



del ddid

del ddtestid

gc.collect()



dd.head()
model_feats = [

    'card1', 'card2', 'card5', 'addr1', 'id_19', 'id_20',

    'dist1', 'P_emaildomain',

    'C1', 'C2', 'C5', 'C6', 'C9', 'C11', 'C13', 'C14',

    'D1', 'D2', 'D3', 'D4', 'D5', 'D8', 'D10', 'D11', 'D15',

    'id_01', 'id_02', 'id_05', 'id_30',

    'V45', 'V54', 'V62', 'V67', 'V83', 'V87', 'V100', 'V129', 'V135', 'V169', 'V313'

    ]

      

for col in ["card1", "card2", "card3", "card5", "card6", "addr1", "P_emaildomain", "R_emaildomain", "TransactionAmt", "id_19", "id_20", "id_31", "id_33", "M5", "M6", "DeviceInfo"]:

    mapping = pd.concat([dd[col], ddtest[col]]).value_counts()

    col_name = f"cnts_{col}"

    

    dd[col_name] = dd[col].map(mapping)

    ddtest[col_name] = ddtest[col].map(mapping)

    

    model_feats.append(col_name)

    

for col in ["ProductCD", "card1", "card2", "card3", "card4", "card5", "addr2", "DeviceType"]:

    gr = dd.groupby(col)["TransactionAmt"]

    mean = gr.transform("mean")

    std = gr.transform("std")

    

    col_name = f"amount_score_{col}"

    dd[col_name] = (dd["TransactionAmt"] - mean) / std

    ddtest[col_name] = (ddtest["TransactionAmt"] - mean) / std

    

    model_feats.append(col_name)

    

for col in ["P_emaildomain", "id_30"]:

    dd[col] = dd[col].astype("category")

    ddtest[col] = ddtest[col].astype("category")

 

len(model_feats)
from math import sqrt

from sklearn.base import clone

from sklearn.model_selection import cross_validate

import numpy as np

from operator import itemgetter

from functools import partial

import time





class Dummy:

    def __getattr__(self, name):

        return lambda x:x

    

colorful=Dummy()     # should be library from `pip install colorful` but didnt work





def format_if_number(x, color=None, format="{:g}"):

    if isinstance(x, (int, float)):

        text = format.format(x)

    else:

        text = x



    if color is not None:

        text = color(text)



    return text



color_score = partial(format_if_number, color=colorful.violet)

color_number = partial(format_if_number, color=colorful.cornflowerBlue, format="{}")

color_param_val = partial(format_if_number, color=colorful.deepSkyBlue)

color_param_name = partial(format_if_number, color=colorful.limeGreen, format="{}")





class SearchStop(Exception):

    pass





class GoldenSearch:

    """

    def func(x):

        return x**2



    g=GoldenSearch(-10, 10)

    gen=g.val_gen()

    y=None

    try:

        while g.c-g.a>0.1:

            x = gen.send(y)

            y = func(x)

            print(g)

    except SearchStop as exc:

        pass

    """



    pos = 2 - (1 + sqrt(5)) / 2  # ~0.382



    def __init__(

        self,

        x0,

        x1,

        y0=np.nan,

        y1=np.nan,

        *,

        xm=np.nan,

        ym=np.nan,

        min_bound=True,

        max_bound=True,

        noise=0,

        map_value=None,

    ):

        if map_value is None:

            map_value = lambda x: x

        self.map_value = map_value



        self.a = map_value(x0)

        self.c = map_value(x1)



        if np.isnan(xm):

            xm = map_value(self.a + self.pos * (self.c - self.a))

        self.b = xm



        if min_bound is True:

            self.min_bound = self.a

        else:

            self.min_bound = min_bound



        if max_bound is True:

            self.max_bound = self.c

        else:

            self.max_bound = max_bound



        self.noise = noise



        self.ya = y0

        self.yb = ym

        self.yc = y1



        self.new_x = np.nan

        self.new_y = np.nan



    def _map_value(self, value):

        value = self.map_value(value)

        if value == self.a or value == self.b or value == self.c:

            raise SearchStop(f"Repeated value {value}")



        return value



    def val_gen(self):

        if np.isnan(self.ya):

            self.ya = yield self.a



        if np.isnan(self.yc):

            self.yc = yield self.c



        if np.isnan(self.yb):

            self.yb = yield self.b



        while 1:

            d1 = self.b - self.a

            d2 = self.c - self.b



            if self.ya < self.yb <= self.yc + self.noise:  # extend region left

                if self.min_bound < self.a:

                    self.new_x = self._map_value(self.b - d1 / self.pos)



                    if self.new_x < self.min_bound:

                        self.new_x = self.min_bound



                    self.new_y = yield self.new_x

                    self.a, self.b, self.c = self.new_x, self.a, self.b

                    self.ya, self.yb, self.yc = self.new_y, self.ya, self.yb

                else:

                    self.new_x = self._map_value(self.a + self.pos * (self.b - self.a))



                    self.new_y = yield self.new_x

                    self.b, self.c = self.new_x, self.b

                    self.yb, self.yc = self.new_y, self.yb



            elif self.ya + self.noise >= self.yb > self.yc:  # extend region right

                if self.c < self.max_bound:

                    self.new_x = self._map_value(self.b + d2 / self.pos)



                    if self.new_x > self.max_bound:

                        self.new_x = self.max_bound



                    self.new_y = yield self.new_x

                    self.a, self.b, self.c = self.b, self.c, self.new_x

                    self.ya, self.yb, self.yc = self.yb, self.yc, self.new_y

                else:

                    self.new_x = self._map_value(self.c - self.pos * (self.c - self.b))



                    self.new_y = yield self.new_x

                    self.a, self.b = self.b, self.new_x

                    self.ya, self.yb = self.yb, self.new_y



            elif self.ya >= self.yb - self.noise and self.yb - self.noise <= self.yc:

                if d1 < d2:

                    self.new_x = self._map_value(

                        self.c - (1 - self.pos) * (self.c - self.b)

                    )



                    self.new_y = yield self.new_x



                    if self.new_y > self.yc + self.noise:

                        raise SearchStop("Inconsistent y > c")



                    if self.new_y < self.yb:

                        self.a, self.b = self.b, self.new_x

                        self.ya, self.yb = self.yb, self.new_y

                    elif self.new_y > self.yb:

                        self.c = self.new_x

                        self.yc = self.new_y

                    else:

                        raise SearchStop("Inconsistent y = c")

                else:

                    self.new_x = self._map_value(

                        self.a + (1 - self.pos) * (self.b - self.a)

                    )



                    self.new_y = yield self.new_x



                    if self.new_y > self.ya + self.noise:

                        raise SearchStop("Inconsistent y > a")



                    if self.new_y < self.yb:

                        self.b, self.c = self.new_x, self.b

                        self.yb, self.yc = self.new_y, self.yb

                    elif self.new_y > self.yb:

                        self.a = self.new_x

                        self.ya = self.new_y

                    else:

                        raise SearchStop("Inconsistent y = b")

            else:

                raise SearchStop("Inconsistent a < b > c")



    def __repr__(self):

        vals = [

            (self.a, self.ya),

            (self.b, self.yb),

            (self.c, self.yc),

            (self.new_x, np.nan),

        ]

        vals.sort(key=itemgetter(0))



        format_if_not_nan = lambda x: f"{x:g}" if not np.isnan(x) else "_"



        return (

            f"Golden( "

            + " | ".join(

                f"{format_if_not_nan(x)}:{format_if_not_nan(y)}" for x, y in vals

            )

            + f" -> {min(self.ya, self.yb, self.yc):g} )"

        )





class GoldenSearcher:

    def __init__(

        self, param_name, target_precision, x0, x1, *golden_args, **golden_kwargs

    ):

        self.param_name = param_name

        self.target_precision = target_precision



        if (

            "map_value" not in golden_kwargs

            and isinstance(x0, int)

            and isinstance(x1, int)

        ):

            golden_kwargs["map_value"] = int



        self.searcher = GoldenSearch(x0, x1, *golden_args, **golden_kwargs)

        self.val_gen = self.searcher.val_gen()



    def next_search_params(self, params, last_score):

        val = self.val_gen.send(last_score)



        if self.searcher.c - self.searcher.a < self.target_precision:

            raise SearchStop(f"Target precision {self.target_precision} reached")



        new_params = params.copy()

        new_params[self.param_name] = val

        return new_params



    def state_info(self):

        return str(self.searcher)



    def __repr__(self):

        return f"GoldenSearcher({self.param_name})"





class ListSearcher:

    def __init__(self, param_name, val_list):

        self.param_name = param_name

        self.val_list = val_list

        self.idx = -1



    def next_search_params(self, params, last_score):

        self.idx += 1



        if self.idx == len(self.val_list):

            raise SearchStop(f"Last of {len(self.val_list)} list values reached")



        new_params = params.copy()

        new_val = self.val_list[self.idx]

        new_params[self.param_name] = new_val



        return new_params



    def state_info(self):

        return f"ListSearcher(val {self.idx+1}/{len(self.val_list)})"



    def __repr__(self):

        return f"ListSearcher({self.param_name}, {len(self.val_list)} vals)"





class SearcherCV:

    def __init__(self, estimator, searchers, *, scoring, cv, num_feat_imps=5):

        self.estimator = estimator



        self.searchers = searchers



        self.scoring = scoring

        self.cv = cv

        self.best_params_ = None

        self.best_score_ = None



        self.num_feat_imps = num_feat_imps



    def fit(self, X, y, verbose_search=True, **fit_params):

        if verbose_search:

            print(

                f"Starting fit on {len(X.columns)} features and {len(X)} instances with folds {self.cv} and scoring {self.scoring}"

            )

            print()



        self.best_params_ = {}



        for searcher in self.searchers:

            cur_params = self.best_params_



            if verbose_search:

                print(f">> Starting searcher {searcher}")



            try:

                score = None

                while 1:  # SearchStop expected

                    cur_params = searcher.next_search_params(cur_params, score)



                    mark = (

                        lambda param_name: "*"

                        if hasattr(searcher, "param_name")

                        and searcher.param_name == param_name

                        else ""

                    )



                    new_params_str = ", ".join(

                        f"{mark(param_name)}{param_name}{mark(param_name)} = {format_if_number(param_val)}"

                        for param_name, param_val in sorted(cur_params.items())

                    )



                    if verbose_search:

                        print()

                        print(

                            f"Current best score:",

                            color_score(self.best_score_)

                            if self.best_score_ is not None

                            else "-",

                        )

                        print(

                            f"Current best params:",

                            ", ".join(

                                f"{param}={format_if_number(val)}"

                                for param, val in sorted(self.best_params_.items())

                            )

                            if self.best_params_ is not None

                            else "-",

                        )

                        print(f"Searcher state: {searcher.state_info()}")

                        print(f"-> Eval: {new_params_str} .......")



                    start_time = time.time()

                    score = self._score(X, y, cur_params, fit_params)

                    end_time = time.time()

                    run_time_min = (end_time - start_time) / 60



                    new_params_color_str = ", ".join(

                        f"{color_param_name(param_name)} = {color_param_val(param_val)}"

                        if hasattr(searcher, "param_name")

                        and searcher.param_name == param_name

                        else f"{param_name} = {format_if_number(param_val)}"

                        for param_name, param_val in sorted(cur_params.items())

                    )



                    print(

                        f"....... ({run_time_min:.2g}min) {new_params_color_str} >>> {color_score(score)}"

                    )



                    if self.best_score_ is None or score < self.best_score_:  #!!!

                        self.best_score_ = score

                        self.best_params_ = cur_params.copy()



            except SearchStop as exc:

                print()

                print(f"Searcher {searcher} stopped with: {exc}")

                print()



        if verbose_search:

            print(f"Final best score: {color_score(self.best_score_)}")

            print(f"Final best params:")

            for param, val in sorted(self.best_params_.items()):

                print(f"    {color_param_name(param)} = {color_param_val(val)},")



    def _score(self, X, y, params, fit_params):

        estimator = clone(self.estimator)

        estimator.set_params(**params)



        cross_val_info = cross_validate(

            estimator,

            X,

            y,

            scoring=self.scoring,

            cv=self.cv,

            fit_params=fit_params,

            return_train_score=True,

            return_estimator=True,

        )



        for fold_idx, (clf, train_score, test_score) in enumerate(

            zip(

                cross_val_info["estimator"],

                cross_val_info["train_score"],

                cross_val_info["test_score"],

            ),

            1,

        ):

            infos = [f"{test_score:g} (train {train_score:g})"]

            if hasattr(clf, "best_iteration_") and clf.best_iteration_ is not None:

                infos.append(f"best iter {clf.best_iteration_}")



            if hasattr(clf, "best_score_") and clf.best_score_:

                if isinstance(clf.best_score_, dict):

                    best_score_str = ", ".join(

                        (f"{set_name}(" if len(clf.best_score_) > 1 else "")

                        + ", ".join(

                            f"{score_name}={score:g}"

                            for score_name, score in scores.items()

                        )

                        + (")" if len(clf.best_score_) > 1 else "")

                        for set_name, scores in clf.best_score_.items()

                    )

                else:

                    best_score_str = format_if_number(clf.best_score_)

                    

                infos.append(f"stop score {best_score_str}")



            if hasattr(clf, "feature_importances_"):

                feat_imps = sorted(

                    zip(clf.feature_importances_, X.columns), reverse=True

                )

                infos.append(

                    "Top feat: "

                    + " · ".join(

                        str(feat) for _score, feat in feat_imps[: self.num_feat_imps]

                    )

                )



            print(f"Fold {fold_idx}:", "; ".join(infos))



        score = cross_val_info["test_score"].mean()



        return -score



    

def earlystop(

    clf,

    X,

    y,

    *,

    eval_metric=None,

    early_stopping_rounds=100,

    test_size=0.1,

    verbose=False,

    **fit_params,

):

    X_train, X_stop, y_train, y_stop = train_test_split(X, y, test_size=test_size)



    clf.fit(

        X_train,

        y_train,

        early_stopping_rounds=early_stopping_rounds,

        eval_set=[(X_stop, y_stop)],

        eval_metric=eval_metric,

        verbose=verbose,

        **fit_params,

    )



    infos = []

    if hasattr(clf, "best_iteration_") and clf.best_iteration_ is not None:

        infos.append(f"Best iter {clf.best_iteration_}")



        if hasattr(clf, "best_score_") and clf.best_score_:

            if isinstance(clf.best_score_, dict):

                best_score_str = ", ".join(

                    (f"{set_name}(" if len(clf.best_score_) > 1 else "")

                    + ", ".join(

                        f"{score_name}={score:g}" for score_name, score in scores.items()

                    )

                    + (")" if len(clf.best_score_) > 1 else "")

                    for set_name, scores in clf.best_score_.items()

                )

            else:

                best_score_str = format_if_number(clf.best_score_)



            infos.append(f"Stop scores {best_score_str}")



    if hasattr(clf, "feature_importances_"):

        feat_imps = sorted(zip(clf.feature_importances_, X.columns), reverse=True)

        infos.append(

            "Top feat: "

            + " · ".join(feat for _score, feat in feat_imps[: self.num_feat_imps])

        )

    print("\n".join(infos))
cv = KFold(3)



base_params=dict(

    learning_rate=0.3,  # high learning rate because didnt manage to install GPU

    #device_type="gpu",

    #max_bin=63,

)



clf = LGBMClassifier(

    n_estimators=5000,

    colsample_bytree=0.9,  # let's start with 0.9, but will be optimized later

    **base_params,

)



X = dd[model_feats]   # it's only a few features for demo, so it won't perform well

y = dd["isFraud"]



search = SearcherCV(

    clf,

    [

        GoldenSearcher("num_leaves", 30, 200, 800, noise=0.003),   # target_precision 30, try values between 200 and 800, noise makes the searcher not stop when it the results do not look unimodal

        ListSearcher("colsample_bytree", [0.3, 0.5, 0.7, 0.9]),

    ],

    scoring="roc_auc",

    cv=cv,

)



earlystop(search, X, y, eval_metric="auc")
clf = LGBMClassifier(

    **{**base_params,

    **search.best_params_}

  )



clf.fit(X, y)



preds = clf.predict_proba(ddtest[model_feats])[:, 1]



sub = pd.DataFrame({"TransactionID": ddtest["TransactionID"], "isFraud": preds})

sub.to_csv("submission.csv.gz", index=False)