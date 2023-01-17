import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
# Loading data directly from CatBoost

from catboost.datasets import amazon

train, test = amazon()

target = "ACTION"

col4train = [x for x in train.columns if x not in [target, "ROLE_TITLE"]]

y = train[target].values
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from sklearn.ensemble import ExtraTreesClassifier



#our small helper function, returns ExtraTrees instance

def get_model():

    params = {

        "n_estimators":300, 

        "n_jobs": 3,

        "random_state":5436,

    }

    return ExtraTreesClassifier(**params)
from sklearn.base import BaseEstimator, TransformerMixin

class TargetEncoding(BaseEstimator, TransformerMixin):

    def __init__(self, columns_names ):

        self.columns_names = columns_names

        self.learned_values = {}

        self.dataset_mean = np.nan

    

    def fit(self, X, y, **fit_params):

        X_ = X.copy()

        self.learned_values = {}

        X_["__target__"] = y

        for c in [x for x in X_.columns if x in self.columns_names]:

            self.learned_values[c] = (X_[[c,"__target__"]]

                                      .groupby(c)["__target__"].mean()

                                      .reset_index())

        self.dataset_mean = np.mean(y)

        return self

    

    def transform(self, X, **fit_params):

        transformed_X = X[self.columns_names].copy()

        for c in transformed_X.columns:

            transformed_X[c] = (transformed_X[[c]]

                                .merge(self.learned_values[c], on = c, how = 'left')

                               )["__target__"]

        transformed_X = transformed_X.fillna(self.dataset_mean)

        return transformed_X

    

    def fit_transform(self, X, y, **fit_params):

        self.fit(X,y)

        return self.transform(X)
skf = StratifiedKFold(n_splits=5, random_state = 5451, shuffle = True)

te = TargetEncoding(columns_names=col4train)

X_tr = te.fit_transform(train, y).values



scores = []

tr_scores = []

for train_index, test_index in skf.split(train, y):

    train_df, valid_df = X_tr[train_index], X_tr[test_index]

    train_y, valid_y = y[train_index], y[test_index]



    model = get_model()

    model.fit(train_df,train_y)



    predictions = model.predict_proba(valid_df)[:,1]

    scores.append(roc_auc_score(valid_y, predictions))



    train_preds = model.predict_proba(train_df)[:,1]

    tr_scores.append(roc_auc_score(train_y, train_preds))



print("Train AUC score: {:.4f} Valid AUC score: {:.4f}, STD: {:.4f}".format(

    np.mean(tr_scores), np.mean(scores), np.std(scores)

))
scores = []

tr_scores = []

for train_index, test_index in skf.split(train, y):

    train_df = train.loc[train_index,col4train].reset_index(drop = True)

    valid_df = train.loc[test_index,col4train].reset_index(drop = True)

    train_y, valid_y = y[train_index], y[test_index]

    

    te = TargetEncoding(columns_names=col4train)

    X_tr = te.fit_transform(train_df, train_y).values

    X_val = te.transform(valid_df).values



    model = get_model()

    model.fit(X_tr,train_y)



    predictions = model.predict_proba(X_val)[:,1]

    scores.append(roc_auc_score(valid_y, predictions))



    train_preds = model.predict_proba(X_tr)[:,1]

    tr_scores.append(roc_auc_score(train_y, train_preds))



print("Train AUC score: {:.4f} Valid AUC score: {:.4f}, STD: {:.4f}".format(

    np.mean(tr_scores), np.mean(scores), np.std(scores)

))
class TargetEncodingSmoothing(BaseEstimator, TransformerMixin):

    def __init__(self, columns_names,k, f ):

        self.columns_names = columns_names

        self.learned_values = {}

        self.dataset_mean = np.nan

        self.k = k #

        self.f = f #

    def smoothing_func(self, N): #

        return 1 / (1 + np.exp(-(N-self.k)/self.f))

    def fit(self, X, y, **fit_params):

        X_ = X.copy()

        self.learned_values = {}

        self.dataset_mean = np.mean(y)

        X_["__target__"] = y

        for c in [x for x in X_.columns if x in self.columns_names]:

            stats = (X_[[c,"__target__"]]

                     .groupby(c)["__target__"].

                     agg(['mean', 'size'])) 

            stats["alpha"] = self.smoothing_func(stats["size"])

            stats["__target__"] = (stats["alpha"]*stats["mean"] 

                                   + (1-stats["alpha"])*self.dataset_mean)

            stats = (stats

                     .drop([x for x in stats.columns if x not in ["__target__",c]], axis = 1)

                     .reset_index())

            self.learned_values[c] = stats

        self.dataset_mean = np.mean(y)

        return self

    def transform(self, X, **fit_params):

        transformed_X = X[self.columns_names].copy()

        for c in transformed_X.columns:

            transformed_X[c] = (transformed_X[[c]]

                                .merge(self.learned_values[c], on = c, how = 'left')

                               )["__target__"]

        transformed_X = transformed_X.fillna(self.dataset_mean)

        return transformed_X

    def fit_transform(self, X, y, **fit_params):

        self.fit(X,y)

        return self.transform(X)
%matplotlib inline

x = np.linspace(0,100,100)

plot = pd.DataFrame()

te = TargetEncodingSmoothing([], 1,1)

plot["k=1|f=1"] = te.smoothing_func(x)

te = TargetEncodingSmoothing([], 33,5)

plot["k=33|f=5"] = te.smoothing_func(x)

te = TargetEncodingSmoothing([], 66,15)

plot["k=66|f=15"] = te.smoothing_func(x)

plot.plot(figsize = (15,8))
scores = []

tr_scores = []

for train_index, test_index in skf.split(train, y):

    train_df = train.loc[train_index,col4train].reset_index(drop = True)

    valid_df = train.loc[test_index,col4train].reset_index(drop = True)

    train_y, valid_y = y[train_index], y[test_index]

    te = TargetEncodingSmoothing(

        columns_names= col4train,

        k = 3, f = 1.5

    )

    X_tr = te.fit_transform(train_df, train_y).values

    X_val = te.transform(valid_df).values



    model = get_model()

    model.fit(X_tr,train_y)



    predictions = model.predict_proba(X_val)[:,1]

    scores.append(roc_auc_score(valid_y, predictions))



    train_preds = model.predict_proba(X_tr)[:,1]

    tr_scores.append(roc_auc_score(train_y, train_preds))



print("Train AUC score: {:.4f} Valid AUC score: {:.4f}, STD: {:.4f}".format(

    np.mean(tr_scores), np.mean(scores), np.std(scores)

))
def get_CV_target_encoding(data, y, encoder, cv = 5):

    skfTE = StratifiedKFold(n_splits=cv, random_state = 545167, shuffle = True)

    result = []

    for train_indexTE, test_indexTE in skfTE.split(data, y):

        encoder.fit(data.iloc[train_indexTE,:].reset_index(drop = True), y[train_indexTE])

        tmp =  encoder.transform(data.iloc[test_indexTE,:].reset_index(drop = True))

        tmp["index"] = test_indexTE

        result.append(tmp)

    result = pd.concat(result, ignore_index = True)

    result = result.sort_values('index').reset_index(drop = True).drop('index', axis = 1)

    return result
scores = []

tr_scores = []

for train_index, test_index in skf.split(train, y):

    train_df = train.loc[train_index,col4train].reset_index(drop = True)

    valid_df = train.loc[test_index,col4train].reset_index(drop = True)

    train_y, valid_y = y[train_index], y[test_index]

    te = TargetEncodingSmoothing(

        columns_names= col4train,

        k = 3, f = 1.5

    )

    

    X_tr = get_CV_target_encoding(train_df, train_y, te, cv = 5)



    te.fit(train_df, train_y)

    X_val = te.transform(valid_df).values



    model = get_model()

    model.fit(X_tr,train_y)



    predictions = model.predict_proba(X_val)[:,1]

    scores.append(roc_auc_score(valid_y, predictions))



    train_preds = model.predict_proba(X_tr)[:,1]

    tr_scores.append(roc_auc_score(train_y, train_preds))



print("Train AUC score: {:.4f} Valid AUC score: {:.4f}, STD: {:.4f}".format(

    np.mean(tr_scores), np.mean(scores), np.std(scores)

))
class TargetEncodingExpandingMean(BaseEstimator, TransformerMixin):

    def __init__(self, columns_names):

        self.columns_names = columns_names

        self.learned_values = {}

        self.dataset_mean = np.nan

    def fit(self, X, y, **fit_params):

        X_ = X.copy()

        self.learned_values = {}

        self.dataset_mean = np.mean(y)

        X_["__target__"] = y

        for c in [x for x in X_.columns if x in self.columns_names]:

            stats = (X_[[c,"__target__"]]

                     .groupby(c)["__target__"]

                     .agg(['mean', 'size'])) #

            stats["__target__"] = stats["mean"]

            stats = (stats

                     .drop([x for x in stats.columns if x not in ["__target__",c]], axis = 1)

                     .reset_index())

            self.learned_values[c] = stats

        return self

    def transform(self, X, **fit_params):

        transformed_X = X[self.columns_names].copy()

        for c in transformed_X.columns:

            transformed_X[c] = (transformed_X[[c]]

                                .merge(self.learned_values[c], on = c, how = 'left')

                               )["__target__"]

        transformed_X = transformed_X.fillna(self.dataset_mean)

        return transformed_X

    

    def fit_transform(self, X, y, **fit_params):

        self.fit(X,y)

    

        #Expanding mean transform

        X_ = X[self.columns_names].copy().reset_index(drop = True)

        X_["__target__"] = y

        X_["index"] = X_.index

        X_transformed = pd.DataFrame()

        for c in self.columns_names:

            X_shuffled = X_[[c,"__target__", "index"]].copy()

            X_shuffled = X_shuffled.sample(n = len(X_shuffled),replace=False)

            X_shuffled["cnt"] = 1

            X_shuffled["cumsum"] = (X_shuffled

                                    .groupby(c,sort=False)['__target__']

                                    .apply(lambda x : x.shift().cumsum()))

            X_shuffled["cumcnt"] = (X_shuffled

                                    .groupby(c,sort=False)['cnt']

                                    .apply(lambda x : x.shift().cumsum()))

            X_shuffled["encoded"] = X_shuffled["cumsum"] / X_shuffled["cumcnt"]

            X_shuffled["encoded"] = X_shuffled["encoded"].fillna(self.dataset_mean)

            X_transformed[c] = X_shuffled.sort_values("index")["encoded"].values

        return X_transformed
scores = []

tr_scores = []

for train_index, test_index in skf.split(train, y):

    train_df = train.loc[train_index,col4train].reset_index(drop = True)

    valid_df = train.loc[test_index,col4train].reset_index(drop = True)

    train_y, valid_y = y[train_index], y[test_index]

    te = TargetEncodingExpandingMean(columns_names=col4train)



    X_tr = te.fit_transform(train_df, train_y)

    X_val = te.transform(valid_df).values



    model = get_model()

    model.fit(X_tr,train_y)



    predictions = model.predict_proba(X_val)[:,1]

    scores.append(roc_auc_score(valid_y, predictions))



    train_preds = model.predict_proba(X_tr)[:,1]

    tr_scores.append(roc_auc_score(train_y, train_preds))



print("Train AUC score: {:.4f} Valid AUC score: {:.4f}, STD: {:.4f}".format(

    np.mean(tr_scores), np.mean(scores), np.std(scores)

))
train[col4train] = train[col4train].values.astype(str)

test[col4train] = test[col4train].values.astype(str)



from itertools import combinations

new_col4train = col4train

for c1,c2 in combinations(col4train, 2):

    name = "{}_{}".format(c1,c2)

    new_col4train.append(name)

    train[name] = train[c1] + "_" + train[c2]

    test[name] = test[c1] + "_" + test[c2]
print(train[new_col4train].shape, test[new_col4train].shape)

train[new_col4train].head(5)
train[new_col4train].apply(lambda x: len(x.unique()))
scores = []

tr_scores = []

for train_index, test_index in skf.split(train, y):

    train_df = train.loc[train_index,new_col4train].reset_index(drop = True)

    valid_df = train.loc[test_index,new_col4train].reset_index(drop = True)

    train_y, valid_y = y[train_index], y[test_index]

    te = TargetEncodingExpandingMean(columns_names=new_col4train)



    X_tr = te.fit_transform(train_df, train_y)

    X_val = te.transform(valid_df)

    

    te2 = TargetEncodingSmoothing(

        columns_names= new_col4train,

        k = 3, f = 1.5,

    )

    

    X_tr2 = get_CV_target_encoding(train_df, train_y, te2, cv = 5)

    te2.fit(train_df, train_y)

    X_val2 = te2.transform(valid_df)

    

    X_tr = pd.concat([X_tr, X_tr2], axis = 1)

    X_val = pd.concat([X_val, X_val2], axis = 1)



    model = get_model()

    model.fit(X_tr,train_y)



    predictions = model.predict_proba(X_val)[:,1]

    scores.append(roc_auc_score(valid_y, predictions))



    train_preds = model.predict_proba(X_tr)[:,1]

    tr_scores.append(roc_auc_score(train_y, train_preds))



print("Train AUC score: {:.4f} Valid AUC score: {:.4f}, STD: {:.4f}".format(

    np.mean(tr_scores), np.mean(scores), np.std(scores)

))
te = TargetEncodingExpandingMean(columns_names=new_col4train)



X_tr = te.fit_transform(train[new_col4train], y)

X_val = te.transform(test[new_col4train])



te2 = TargetEncodingSmoothing(

    columns_names= new_col4train,

    k = 3, f = 1.5,

)



X_tr2 = get_CV_target_encoding(train[new_col4train], y, te2, cv = 5)

te2.fit(train[new_col4train], y)

X_val2 = te2.transform(test[new_col4train])



X = pd.concat([X_tr, X_tr2], axis = 1)

X_te = pd.concat([X_val, X_val2], axis = 1)



model = get_model()

model.fit(X,y)

predictions = model.predict_proba(X_te)[:,1]



submit = pd.DataFrame()

submit["Id"] = test["id"]

submit["ACTION"] = predictions



submit.to_csv("submission.csv", index = False)