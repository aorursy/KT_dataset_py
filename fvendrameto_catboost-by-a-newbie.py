import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
df_train = pd.read_csv('../input/train.csv')
df_train = pd.DataFrame(df_train)
df_test = pd.read_csv('../input/test.csv')
df_test = pd.DataFrame(df_test)
target = 'winPlacePerc'
train_columns = list(df_train.columns)
train_columns.remove(target)

X = df_train[train_columns]
y = df_train[target]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
def run_catboost(X_train, y_train, X_val, y_val, X_test):
    model = CatBoostRegressor(iterations=1000,
                             learning_rate=0.05,
                             depth=12,
                             eval_metric='MAE',
                             random_seed = 42,
                             od_type='Iter',
                             metric_period = 50,
                             od_wait=20)
    model.fit(X_train, y_train,
              eval_set=(X_val, y_val),
              use_best_model=True,
              verbose=True)
    
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_submit = model.predict(X_test)

    return y_pred_submit
cat_preds = run_catboost(X_train, y_train, X_val, y_val, df_test)
df_test['winPlacePerc'] = cat_preds
submission = df_test[['Id', 'winPlacePerc']]
submission.to_csv('submission.csv', index=False)