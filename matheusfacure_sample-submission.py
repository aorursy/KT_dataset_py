import pandas as pd

import numpy as np
dataset = pd.read_csv("/kaggle/input/fia-machine-learning-t7/credit_train_label.csv")
dataset.head()
## Data Types

dataset.dtypes
## Missing

dataset.isna().mean()
dataset.describe()
# sample size

dataset.shape
target = "SeriousDlqin2yrs"

features = dataset.drop(columns=["X", target]).columns
def feature_eng(df: pd.DataFrame, features):

    return df.assign(**{

        f"{f1}_over_{f1}": df[f1]/df[f1] for f1 in features for f2 in features

    }).assign(**{

        f"{f1}_times_{f1}": df[f1]*df[f1] for f1 in features for f2 in features

    })



eng_dataset = feature_eng(dataset, features)

eng_features = eng_dataset.drop(columns=["X", target]).columns



print(eng_features)

eng_dataset.head()
import xgboost as xgb



from scipy.stats import randint as sp_randint

from sklearn.model_selection import RandomizedSearchCV
param_grid = {

        'silent': [True],

        'max_depth': list(range(2, 11)),

        'learning_rate': [0.001, 0.01, 0.05, 0.08, 0.1, 0.2, 0.3],

        'subsample': [0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0],

        'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],

        'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],

        'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],

        'gamma': [0, 0.25, 0.5, 1.0],

        'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],

        'n_estimators': [50, 100, 200]}



# run randomized search

clf = xgb.XGBClassifier()



n_iter_search = 10

random_search = RandomizedSearchCV(clf, param_distributions=param_grid,

                                   n_iter=n_iter_search, cv=5, iid=False, n_jobs=-1)
random_search.fit(eng_dataset[eng_features], eng_dataset[target])
trained_model = random_search.best_estimator_

print("Best score", random_search.best_score_)

trained_model
test_data = pd.read_csv("/kaggle/input/fia-machine-learning-t7/credit_test_features.csv")

test_data.head()
predictions = trained_model.predict_proba(feature_eng(test_data, features)[eng_features])[:, 1]

predictions.mean()
predictions_df = pd.DataFrame({

    "X": test_data["X"],

    target: predictions 

})



predictions_df.head()