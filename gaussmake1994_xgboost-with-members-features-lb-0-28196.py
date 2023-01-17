%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import xgboost as xgb
train = pd.read_csv("../input/train.csv").merge(

    pd.read_csv("../input/members.csv"),

    on="msno"

)

test = pd.read_csv("../input/sample_submission_zero.csv").merge(

    pd.read_csv("../input/members.csv"),

    on="msno",

    how="left"

)
train.head()
train.describe(include="all")
train.corr()
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import log_loss



def cv(model, X, y, predictor, random_state=None):

    kfold = StratifiedKFold(shuffle=True,

                            random_state=random_state)

    initial_params = model.get_params()

    losses = []

    for i, indices in enumerate(kfold.split(X, y)):

        print("Fold {0}".format(i + 1))

        train_index, test_index = indices

        model.set_params(**initial_params)

        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)

        p_predicted = predictor(model, X_test)

        p_predicted[p_predicted > 1-10**(-15)] = 1-10**(-15)

        p_predicted[p_predicted < 10**(-15)] = 10**(-15)

        losses.append(log_loss(y_test, p_predicted))

    return np.array(losses)
from sklearn.dummy import DummyRegressor



SEED = 42



cv(DummyRegressor("constant", constant=train["is_churn"].mean()),

   np.zeros([len(train),1]),

   train["is_churn"],

   random_state=SEED,

   predictor=lambda estimator, X: estimator.predict(X))
cv(xgb.XGBClassifier(),

   np.array(train[["registered_via"]]),

   train["is_churn"],

   random_state=SEED,

   predictor=lambda estimator, X: estimator.predict_proba(X)[:, 1])
cv(xgb.XGBClassifier(),

   np.array(train[["registered_via", "city"]]),

   train["is_churn"],

   random_state=SEED,

   predictor=lambda estimator, X: estimator.predict_proba(X)[:, 1])
city_values = list(set(train["city"]))

city_values.sort()



city_features = ["city{0}".format(city) for city in city_values]

for city in city_values:

    train["city{0}".format(city)] = train["city"] == city

    test["city{0}".format(city)] = test["city"] == city



cv(xgb.XGBClassifier(),

   np.array(train[["registered_via"] + city_features]),

   train["is_churn"],

   random_state=SEED,

   predictor=lambda estimator, X: estimator.predict_proba(X)[:, 1])
plt.hist(train["bd"]);
plt.hist(np.log(np.abs(train["bd"]) + 0.001));
train["bd"].min(), train["bd"].max(), train["bd"].mean(), train["bd"].std()
cv(xgb.XGBClassifier(),

   np.array(train[["registered_via", "bd"] + city_features]),

   train["is_churn"],

   random_state=SEED,

   predictor=lambda estimator, X: estimator.predict_proba(X)[:, 1])
def gender(val):

    if val == "male":

        return 1

    elif val == "female":

        return -1

    else:

        return float("NaN")

    

train["gender_converted"] = train["gender"].apply(gender)

test["gender_converted"] = test["gender"].apply(gender)
cv(xgb.XGBClassifier(),

   np.array(train[["registered_via", "bd", "gender_converted"] + city_features]),

   train["is_churn"],

   random_state=SEED,

   predictor=lambda estimator, X: estimator.predict_proba(X)[:, 1])
def datetime_to_unix(dt):

    epoch = pd.to_datetime('1970-01-01')

    return (dt - epoch).total_seconds()





train["registration_init_time_date"] = pd.to_datetime(train["registration_init_time"], format="%Y%m%d")

test["registration_init_time_date"] = pd.to_datetime(test["registration_init_time"], format="%Y%m%d")

train["registration_init_time_unix"] = train["registration_init_time_date"].apply(datetime_to_unix)

test["registration_init_time_unix"] = test["registration_init_time_date"].apply(datetime_to_unix)
cv(xgb.XGBClassifier(),

   np.array(train[["registered_via", "bd", "gender_converted", "registration_init_time_unix"] + 

                  city_features]),

   train["is_churn"],

   random_state=SEED,

   predictor=lambda estimator, X: estimator.predict_proba(X)[:, 1])
train["registration_init_time_year"] = train["registration_init_time_date"].apply(lambda date: date.year)

test["registration_init_time_year"] = test["registration_init_time_date"].apply(lambda date: date.year)

train["registration_init_time_month"] = train["registration_init_time_date"].apply(lambda date: date.month)

test["registration_init_time_month"] = test["registration_init_time_date"].apply(lambda date: date.month)

train["registration_init_time_day"] = train["registration_init_time_date"].apply(lambda date: date.day)

test["registration_init_time_day"] = test["registration_init_time_date"].apply(lambda date: date.day)
cv(xgb.XGBClassifier(),

   np.array(train[["registered_via", "bd", "gender_converted", "registration_init_time_unix",

                   "registration_init_time_year", "registration_init_time_month", "registration_init_time_day"] + 

                  city_features]),

   train["is_churn"],

   random_state=SEED,

   predictor=lambda estimator, X: estimator.predict_proba(X)[:, 1])
from collections import OrderedDict



clf = xgb.XGBClassifier()

clf.fit(

    np.array(train[["registered_via", "bd", "gender_converted", "registration_init_time_unix",

                   "registration_init_time_year", "registration_init_time_month", "registration_init_time_day"] + 

                  city_features]),

    np.array(train["is_churn"])

)

prediction = clf.predict_proba(np.array(test[["registered_via", "bd",

                                              "gender_converted", 

                                              "registration_init_time_unix",

                                              "registration_init_time_year", 

                                              "registration_init_time_month",

                                              "registration_init_time_day"] + 

                                             city_features]))[:, 1]

prediction_df = pd.DataFrame(OrderedDict([ ("msno", test["msno"]), ("is_churn", prediction) ]))

prediction_df.head()
prediction_df.to_csv("prediction.csv", index=False)
train["expiration_date"] = pd.to_datetime(train["expiration_date"], format="%Y%m%d")

test["expiration_date"] = pd.to_datetime(test["expiration_date"], format="%Y%m%d")
train["expiration_date_unix"] = train["expiration_date"].apply(datetime_to_unix)

test["expiration_date_unix"] = test["expiration_date"].apply(datetime_to_unix)
cv(xgb.XGBClassifier(),

   np.array(train[["registered_via", "bd", "gender_converted", "registration_init_time_unix",

                   "registration_init_time_year", "registration_init_time_month", "registration_init_time_day",

                   "expiration_date_unix"] + 

                  city_features]),

   train["is_churn"],

   random_state=SEED,

   predictor=lambda estimator, X: estimator.predict_proba(X)[:, 1])