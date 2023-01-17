import numpy as np

import pandas as pd



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV



import lightgbm as lgb

from xgboost import XGBRegressor



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv("../input/creditcard.csv")
df.head()
X_train, X_test, y_train, y_test = train_test_split(df.drop('Class', axis=1), df.Class, test_size=0.3, random_state=42)
print('Train shape', X_train.shape)

print('Test shape', X_test.shape)
print('Start training...')

# train

xgb = XGBRegressor(max_depth=3, n_estimators=1000)

xgb.fit(X_train, y_train, 

        eval_set=[(X_test, y_test)], 

        eval_metric='rmse', 

        early_stopping_rounds=20, 

        verbose=False)



gbm = lgb.LGBMRegressor(objective='regression', num_leaves=31,

                        learning_rate=0.05, n_estimators=20)

gbm.fit(X_train, y_train, 

        eval_set=[(X_test, y_test)],

        eval_metric='l1',

        early_stopping_rounds=5)





print('Start predicting...')

# predict

y_pred_xgb = xgb.predict(X_test)

y_pred_gbm = gbm.predict(X_test, num_iteration=gbm.best_iteration)



# eval

print('The rmse of prediction xgboost is:', mean_squared_error(y_test, y_pred_xgb) ** 0.5)

print('The rmse of prediction lightgbm is:', mean_squared_error(y_test, y_pred_gbm) ** 0.5)



print('Calculate feature importances...')



print("xgboost: feature importance")

xgb_fi = xgb.booster().get_fscore()

print(xgb_fi)



print("lightgbm: feature importance")

gbm_fi = dict(zip(X_train.columns.tolist(),gbm.feature_importances_))

print(gbm_fi)
fig =  plt.figure(figsize = (15, 10))

plt.subplot(2, 2, 1)



plt.bar(np.arange(len(xgb_fi)), xgb_fi.values(), align='center')

plt.xticks(np.arange(len(xgb_fi)), xgb_fi.keys(), fontweight='bold', rotation='vertical')

plt.title('xgboost: feature importance', fontsize=15, fontweight='bold')



plt.subplot(2, 2, 2)

plt.bar(np.arange(len(gbm_fi)), gbm_fi.values(), align='center')

plt.xticks(np.arange(len(gbm_fi)), gbm_fi.keys(), fontweight='bold', rotation='vertical')

plt.title('lightgbm: feature importance', fontsize=15, fontweight='bold')