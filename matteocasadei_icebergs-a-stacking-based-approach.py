import pandas as pd

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import Imputer

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.model_selection import *

from sklearn.linear_model import *

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import make_scorer

from sklearn.metrics import precision_score, recall_score, roc_curve, log_loss

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

%matplotlib inline
train = pd.read_json("../input/trainjson/train.json")

validation = pd.read_json("../input/testjson/test.json")

train_copy = train.copy()

validation_copy = validation.copy()
#Define any necessary function to automitize as much as possible data preprocessing

def double_sample(df):

    '''Horizontally flipping images'''

    img_array_1 = df["band_1"].apply(lambda x: np.array(x, dtype=float).reshape(75, 75)[::,::-1].ravel())

    img_array_2 = df["band_2"].apply(lambda x: np.array(x, dtype=float).reshape(75, 75)[::,::-1].ravel())

    return pd.DataFrame({"band_1": img_array_1, "band_2": img_array_2, "id": df["id"], "inc_angle": df["inc_angle"], "is_iceberg": df["is_iceberg"]})

                        

def extract_band(df, col_name):

    '''

        auxiliary functions just to part band_1 and band_2 columns so as to extract individual 

        features, one per single "pixel".

    '''

    t = df[col_name]

    return t.apply(lambda x: pd.Series(x))



def parse_bands(df):

    '''

        parse all bands an extract individual features out of any single received intensity.

    '''

    t = pd.DataFrame()

    for col in ["band_1", "band_2"]:

        t[[col + "_" + str(i) for i in range(75 * 75)]] = extract_band(df, col)

    return t



def prepare_dataframe(df):

    '''

        prepare the whole dataframe for ML

    '''

    t = parse_bands(df)

    if "is_iceberg" in df.columns:

        t["is_iceberg"] = df["is_iceberg"].copy()

    t["inc_angle"] = df["inc_angle"].copy()

    #drop values containing 'na' string

    indexs_to_drop = t[t["inc_angle"] == 'na'].index

    t = t.drop(indexs_to_drop)

    t["inc_angle"] = t["inc_angle"].astype(float).copy()

    return t
train_copy = train_copy.append(double_sample(train_copy), ignore_index=True)
# let's perform actual data preprocessing 

train_parsed = prepare_dataframe(train_copy)

train_set, test_set = train_test_split(train_parsed, test_size=0.2, random_state=77)



# save separately target values and drop corresponding columns from only-features dataframe

train_target = train_parsed["is_iceberg"]

train_set_target = train_set["is_iceberg"]

test_set_target = test_set["is_iceberg"]

train_parsed.drop("is_iceberg", axis=1, inplace=True)

train_set.drop("is_iceberg", axis=1, inplace=True)

test_set.drop("is_iceberg", axis=1, inplace=True)
#define necessary pipelining before performing actual Machine learning (ML)

#in this case only StandardScaling will be performed



class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attr_names):

        self.col_names = attr_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.col_names].values



def preprocessing_pipeline(columns=[]):

    pipe_num = Pipeline([

        ('selector', DataFrameSelector(columns)),

        ('scaler', StandardScaler())

    ])

    return pipe_num
#perform actual pipelining

train_set_prepared = preprocessing_pipeline(train_set.columns.values).fit_transform(train_set)

test_set_prepared = preprocessing_pipeline(test_set.columns.values).fit_transform(test_set)
#prepare folds on training set

from sklearn.model_selection import KFold

kf = KFold(n_splits=5, random_state=77)
models = ["knn", "logistic", "svc"]
def stack_model(model_name, model, kf, train_set, train_target, 

                test_set, test_target,

                train_meta, test_meta):

    #cv 5 folds

    for train_index, test_index in kf.split(train_set):

        train = train_set[train_index]

        test = train_set[test_index]

        target_train = train_target.iloc[train_index]

        model.fit(train, target_train)

        predictions = model.predict_proba(test)

        train_meta[model_name].loc[test_index] = pd.Series(predictions[:,1], index=test_index, dtype=float)

    #finally predict the test set using the whole train set to fit

    model.fit(train_set, train_target)

    predictions = model.predict_proba(test_set)

    #loss = log_loss(test_target, predictions)

    #print("Log loss on validation set for model {}: {} ".format(model_name, loss))

    test_meta[model_name] = pd.Series(predictions[:,1], dtype=float)
train_meta = pd.DataFrame(index=range(0, train_set.shape[0]), columns=models)

test_meta = pd.DataFrame(index=range(0, test_set.shape[0]), columns=models)
from sklearn.neighbors import KNeighborsClassifier
#param_grid = [{'n_neighbors': [3, 5, 7, 11, 13, 15], 'weights':["uniform", "distance"]}]

#grid = GridSearchCV(KNeighborsClassifier(), param_grid, n_jobs=1, cv=kf, scoring="neg_log_loss", verbose=3)

#grid.fit(train_set_prepared, train_set_target)

#grid.best_estimator_
#param_grid = [{'n_neighbors': [15, 17, 19, 21], 'weights':["distance"]}]

#grid = GridSearchCV(KNeighborsClassifier(), param_grid, n_jobs=1, cv=kf, scoring="neg_log_loss", verbose=3)

#grid.fit(train_set_prepared, train_set_target)

#grid.best_estimator_
best_knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',

           metric_params=None, n_jobs=1, n_neighbors=19, p=2,

           weights='distance')



stack_model("knn", best_knn, kf, train_set_prepared, train_set_target, 

                test_set_prepared, test_set_target,

                train_meta, test_meta)
from sklearn.linear_model import LogisticRegression
#param_grid = [{'solver': ["saga"], 'C': [0.1, 1, 10], 'penalty':["l1", "l2"]}]

#grid = GridSearchCV(LogisticRegression(), param_grid, n_jobs=1, cv=kf, scoring="neg_log_loss", verbose=3)

#grid.fit(train_set_prepared, train_set_target)

#grid.best_estimator_


#param_grid = [{'solver': ["saga"], 'C': [0.1, 0.01, 0.001], 'penalty':["l1"]}]

#grid = GridSearchCV(LogisticRegression(), param_grid, n_jobs=1, cv=kf, scoring="neg_log_loss", verbose=3)

#grid.fit(train_set_prepared, train_set_target)

#grid.best_estimator_
#grid.best_score_
best_logreg = LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,

          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,

          penalty='l1', random_state=None, solver='saga', tol=0.0001,

          verbose=0, warm_start=False)



stack_model("logistic", best_logreg, kf, train_set_prepared, train_set_target, 

                test_set_prepared, test_set_target,

                train_meta, test_meta)
#define functions to extract percentiles

def extract_percentile(df, columns, perc):

    cl = df[columns]

    data = cl.as_matrix()

    rows = data.shape[0]

    percentile = np.percentile(data, perc, axis=1)

    return pd.Series(percentile, df.index)



#process all percentiles

def process_df_with_percentiles(df, columns=["band_1", "band_2"], perc=[5, 10, 25, 50, 75, 95, 99]):

    res = pd.DataFrame(index=df.index)

    for col in columns:

        for p in perc:

            res.loc[df.index, col + "_perc" + str(p)] = extract_percentile(df, [col + "_" + str(i) for i in range(0, 75 * 75)], perc=p)

    #keep inc_angle

    res.loc[df.index, "inc_angle"] = df["inc_angle"].reindex(index=res.index)

    return res
train_set_perc = process_df_with_percentiles(train_set)

test_set_perc = process_df_with_percentiles(test_set)
#perform actual pipelining

train_set_perc_prepared = preprocessing_pipeline(train_set_perc.columns.values).fit_transform(train_set_perc)

test_set_perc_prepared = preprocessing_pipeline(test_set_perc.columns.values).fit_transform(test_set_perc)
from sklearn.svm import SVC
#svc = SVC(probability=True)

#param_grid = [{'degree': [3, 5], 'kernel':["poly"], 'C':[ 1, 10, 100]}, {'kernel': ["linear", "rbf"], 'C':[ 1, 10, 100]}]

#grid = GridSearchCV(svc, param_grid, n_jobs=1, cv=kf, scoring="neg_log_loss", verbose=3)

#grid.fit(train_set_perc_prepared, train_set_target)

#grid.best_estimator_
#svc = SVC(probability=True)

#param_grid = [ {'kernel': ["rbf"], 'C':[ 100, 500, 1000, 5000, 10000]}]

#grid = GridSearchCV(svc, param_grid, n_jobs=1, cv=kf, scoring="neg_log_loss", verbose=3)

#grid.fit(train_set_perc_prepared, train_set_target)

#grid.best_estimator_
#grid.best_score_
best_svc = SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,

  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',

  max_iter=-1, probability=True, random_state=None, shrinking=True,

  tol=0.001, verbose=False)



stack_model("svc", best_svc, kf, train_set_perc_prepared, train_set_target, 

                test_set_perc_prepared, test_set_target,

                train_meta, test_meta)
train_meta_copy = train_meta.copy()

train_target_copy = pd.Series(train_set_target.values, index=train_meta_copy.index)

train_median = train_meta_copy.median(axis=1)

train_meta_copy["median"] = train_median
test_meta_copy = test_meta.copy()

test_target_copy = pd.Series(test_set_target.values, index=test_meta_copy.index)

test_median = test_meta_copy.median(axis=1)

test_meta_copy["median"] = test_median
train_meta_copy.iloc[0:5]
test_meta_copy.iloc[0:5]
param_grid = [{'solver': ["saga"], 'C': [0.1, 1, 10], 'penalty':["l1", "l2"]}]

grid = GridSearchCV(LogisticRegression(), param_grid, n_jobs=1, cv=kf, scoring="neg_log_loss", verbose=3)

grid.fit(train_meta_copy, train_target_copy)

grid.best_estimator_
grid.best_score_
grid.best_estimator_.fit(train_meta_copy, train_target_copy)

test_meta_predictions = grid.best_estimator_.predict_proba(test_meta_copy)
log_loss(test_target_copy, test_meta_predictions)
def prepare_validation_dataframe(df):

    '''

        prepare the whole validation dataframe for ML

    '''

    t = parse_bands(df)

    t["inc_angle"] = df["inc_angle"].copy()

    return t
validation_parsed = prepare_validation_dataframe(validation_copy)

validation_set_prepared = preprocessing_pipeline(validation_parsed.columns.values).fit_transform(validation_parsed)



#this is to prepare the whole test set for fitting base and later staked models

train_prepared = preprocessing_pipeline(train_parsed.columns.values).fit_transform(train_parsed)

train_meta = pd.DataFrame(index=range(0, train_parsed.shape[0]), columns=models)

validation_meta = pd.DataFrame(index=range(0, validation_parsed.shape[0]), columns=models)
best_knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',

           metric_params=None, n_jobs=1, n_neighbors=19, p=2,

           weights='distance')



stack_model("knn", best_knn, kf, train_prepared, train_target, 

                validation_set_prepared, pd.DataFrame(),

                train_meta, validation_meta)
best_logreg = LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,

          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,

          penalty='l1', random_state=None, solver='saga', tol=0.0001,

          verbose=0, warm_start=False)



stack_model("logistic", best_logreg, kf, train_prepared, train_target, 

                validation_set_prepared, pd.DataFrame(),

                train_meta, validation_meta)
train_perc = process_df_with_percentiles(train_parsed)

validation_set_perc = process_df_with_percentiles(validation_parsed)
#perform actual pipelining

train_perc_prepared = preprocessing_pipeline(train_perc.columns.values).fit_transform(train_perc)

validation_set_perc_prepared = preprocessing_pipeline(validation_set_perc.columns.values).fit_transform(validation_set_perc)
best_svc = SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,

  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',

  max_iter=-1, probability=True, random_state=None, shrinking=True,

  tol=0.001, verbose=False)



stack_model("svc", best_svc, kf, train_perc_prepared, train_target, 

                validation_set_perc_prepared, pd.DataFrame(),

                train_meta, validation_meta)
train_meta_copy = train_meta.copy()

train_target_copy = pd.Series(train_target.values, index=train_meta_copy.index)

train_median = train_meta_copy.median(axis=1)

train_meta_copy["median"] = train_median
validation_meta_copy = validation_meta.copy()

validation_median = validation_meta_copy.median(axis=1)

validation_meta_copy["median"] = validation_median
#best lr for stacking devised earlier via grid search

stackedlr = LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,

          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,

          penalty='l2', random_state=None, solver='saga', tol=0.0001,

          verbose=0, warm_start=False)



stackedlr.fit(train_meta_copy, train_target_copy)

validation_predictions = stackedlr.predict_proba(validation_meta_copy)
validation_predictions
validation_ids = validation_copy["id"].copy()
submission_df = {"id": validation_ids, "is_iceberg": validation_predictions[:,1]}

submission = pd.DataFrame(submission_df)

submission.to_csv(path_or_buf="submission-stacking.csv",index=False)
