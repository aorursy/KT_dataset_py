import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df["TotalCharges"] = df["TotalCharges"].apply(lambda x: float("0"+x.strip()))

df.head()
from sklearn.preprocessing import LabelEncoder

cat_cols = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", 

            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", 

            "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]

lbl_enc = LabelEncoder()

for col in cat_cols:

    df[col] = lbl_enc.fit_transform(df[col])

df.head()
churn_map_dict = {"Yes":1, "No":0}

df["Churn"] = df["Churn"].map(churn_map_dict)

df["Churn"].value_counts()



cols_to_use = [col for col in df.columns if col not in ["customerID", "Churn"]]

X = df[cols_to_use].values

y = df["Churn"].values
from sklearn.model_selection import train_test_split



train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=2020)
from bayes_opt import BayesianOptimization

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier as RFC



# function for cross validation

def rfc_cv(n_estimators, min_samples_split, max_features, data, targets):

    estimator = RFC(

        n_estimators=n_estimators,

        min_samples_split=min_samples_split,

        max_features=max_features,

        random_state=2

    )

    cval = cross_val_score(estimator, data, targets,

                           scoring='neg_log_loss', cv=4)

    return cval.mean()



# optimization function

def optimize_rfc(data, targets):

    def rfc_crossval(n_estimators, min_samples_split, max_features):

        return rfc_cv(

            n_estimators=int(n_estimators),

            min_samples_split=int(min_samples_split),

            max_features=max(min(max_features, 0.999), 1e-3),

            data=data,

            targets=targets,

        )



    # evaluation space

    optimizer = BayesianOptimization(

        f=rfc_crossval,

        pbounds={

            "n_estimators": (10, 250),

            "min_samples_split": (2, 25),

            "max_features": (0.1, 0.999),

        },

        random_state=1234,

        verbose=2

    )

    optimizer.maximize(n_iter=10)



    print("Final result:", optimizer.max)

    

    return optimizer

    

result = optimize_rfc(train_X, train_y)
print(result.max)
from hyperopt import hp, tpe, STATUS_OK, Trials

from hyperopt.fmin import fmin



space ={

    'n_estimators': hp.choice('n_estimators', np.arange(10, 250, dtype=int)),

    'min_samples_split' : hp.choice('min_samples_split', np.arange(2, 25, dtype=int)),

    'max_features': hp.quniform ('max_features', 0.1, 0.99, 0.02)

    }



def rfc_cv(space, data=train_X, targets=train_y):

    estimator = RFC(

        n_estimators=space["n_estimators"],

        min_samples_split=space["min_samples_split"],

        max_features=space["max_features"],

        random_state=2

    )

    cval = cross_val_score(estimator, data, targets,

                           scoring='neg_log_loss', cv=4)

    return {'loss':-cval.mean(), 'status': STATUS_OK }



trials = Trials()

best = fmin(fn=rfc_cv,

            space=space,

            algo=tpe.suggest,

            max_evals=5, # change

            trials=trials)
print(best)
import h2o

h2o.init()
h2o_train = h2o.H2OFrame(train_X).cbind(h2o.H2OFrame(train_y))

h2o_test = h2o.H2OFrame(test_X).cbind(h2o.H2OFrame(test_y))
from h2o.estimators.gbm import H2OGradientBoostingEstimator



predictors = h2o_train.columns[:-1]

response = "C110"

h2o_train[response] = h2o_train[response].asfactor()



gbm = H2OGradientBoostingEstimator()

gbm.train(x=predictors, y=response, training_frame=h2o_train)
from h2o.automl import H2OAutoML



aml = H2OAutoML(max_models = 10, 

                max_runtime_secs=100, 

                seed = 1,

                stopping_metric = "logloss",

                nfolds=4

               )

aml.train(x=predictors, y=response, training_frame=h2o_train)
aml.leaderboard
aml.leader
import autosklearn.classification



automl = autosklearn.classification.AutoSklearnClassifier(

    time_left_for_this_task=100,

    per_run_time_limit=30,

    resampling_strategy='cv',

    resampling_strategy_arguments={'folds': 4}

)

automl.fit(train_X.copy(), train_y.copy())
from tpot import TPOTClassifier



automl = TPOTClassifier(max_time_mins=1,

                        max_eval_time_mins=0.5, 

                        scoring='neg_log_loss',

                        cv=4,

                        random_state=2020,

                        verbosity=1

                       )

automl.fit(train_X, train_y)
automl.fitted_pipeline_