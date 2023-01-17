import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))

print(os.listdir("../input/lanl-features"))

print(os.listdir("../input/LANL-Earthquake-Prediction"))
train = pd.read_csv("../input/lanl-features/train_features.csv")

test = pd.read_csv("../input/lanl-features/test_features.csv")

y = pd.read_csv("../input/lanl-features/y.csv")
#train.drop('seg_id', axis=1, inplace=True)

#test.drop('seg_id', axis=1, inplace=True)
from catboost import CatBoostRegressor, Pool

from sklearn import model_selection



cat_features = ['var_larger_than_std_dev']

X_train, X_valid, y_train, y_valid = model_selection.train_test_split(train, y, test_size=0.2, shuffle=True)

model = CatBoostRegressor(iterations = 40_000, 

                          loss_function='MAE', 

                          boosting_type='Ordered',

                          use_best_model = True, 

                          verbose = 500, 

                          early_stopping_rounds = 1000,

                          cat_features = cat_features,

                          task_type = "GPU")

model.fit(X_train, y_train, eval_set=(X_valid,y_valid), use_best_model = True, plot=True)
y_pred = model.predict(test)
submission = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})    

submission.time_to_failure = y_pred

submission.to_csv('submission.csv')

submission