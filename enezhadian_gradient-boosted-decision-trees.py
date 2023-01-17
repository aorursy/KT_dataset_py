import numpy as np

import pandas as pd

import xgboost as xgb

from sklearn.preprocessing import normalize

from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.model_selection import StratifiedKFold
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

features = train.columns[1:-1]
splits = 5

skf = StratifiedKFold(n_splits=splits, shuffle=False, random_state=0)



test_dm = xgb.DMatrix(test[features])

test_predictions = np.zeros(test.shape[0], dtype=np.float)



xgb_rounds = 50

xgb_params = {

              "objective": "binary:logistic",

              "eval_metric": "logloss",

              "eta": 5e-2

             }



for i, (train_indices, validation_indices) in enumerate(skf.split(train[features], train.TARGET)):

    print("Iteration", i + 1)

    visible_train = train.iloc[train_indices]

    visible_train_dm = xgb.DMatrix(visible_train[features], visible_train.TARGET)

    

    validation = train.iloc[validation_indices]

    validation_dm = xgb.DMatrix(validation[features], validation.TARGET)



    classifier = xgb.train(xgb_params,

                           visible_train_dm,

                           xgb_rounds,

                           evals=[(validation_dm, "validation")],

                           early_stopping_rounds=10)

    

    predictions = classifier.predict(validation_dm)

    print('Validation Accuray:', accuracy_score(validation.TARGET, np.rint(predictions)))

    print('Validation ROC:', roc_auc_score(validation.TARGET, predictions))

    

    test_predictions += classifier.predict(test_dm)



test_predictions /= splits

predictions = pd.DataFrame({"ID": test.ID, "TARGET": test_predictions})

predictions.to_csv("predictions.csv", index=False)

print('Done.')