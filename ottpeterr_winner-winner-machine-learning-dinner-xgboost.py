import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
X = pd.read_csv("../input/train.csv")
X = X.sample(frac=1) # shuffle
Y = X.winPlacePerc
X = X.drop(columns=['winPlacePerc'])
use_cols = ['Id', 'matchId', 'groupId', 'damageDealt', 'headshotKills', 'heals','killPlace', 'killPoints', 'kills', 'winPoints']
X = X[use_cols]
X.head()
xgb_model = xgb.XGBClassifier(max_depth=3, n_estimators=50, learning_rate=0.05)
subset = 500
xgb_model.fit(X[0:subset], Y[0:subset], verbose=True)
X_test = pd.read_csv("../input/test.csv")
X_test = X_test[use_cols]


Y_pred = xgb_model.predict(X_test)
Y_pred[Y_pred > 1] = 1
Y_pred[Y_pred < 0] = 0
X_test['winPlacePercPredictions'] = Y_pred

aux = X_test.groupby(['matchId','groupId'])['winPlacePercPredictions'].agg('mean').groupby('matchId').rank(pct=True).reset_index()
aux.columns = ['matchId','groupId','winPlacePerc']
X_test = X_test.merge(aux, how='left', on=['matchId','groupId'])
    
submission = X_test[['Id','winPlacePerc']]
submission.to_csv('predictions_xgboost.csv',index=None) 