from xgboost import XGBRegressor, XGBClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV, learning_curve, validation_curve, StratifiedKFold
from time import time
df = pd.read_csv('../input/train.csv')
df.columns
X = df[['assists', 'boosts', 'damageDealt', 'DBNOs',
       'headshotKills', 'heals', 'killPlace', 'killPoints', 'kills',
       'killStreaks', 'longestKill', 'maxPlace', 'numGroups', 'revives',
       'rideDistance', 'roadKills', 'swimDistance', 'teamKills',
       'vehicleDestroys', 'walkDistance', 'weaponsAcquired', 'winPoints']]
y = df['winPlacePerc']
t = time()
cv= KFold(n_splits=5,random_state=17)
xgb = XGBRegressor(random_state=17)
params = {'learning_rate':[0.1],'max_depth':[5],'n_estimators':[500]}
grs = GridSearchCV(estimator=xgb,param_grid=params,cv=cv,n_jobs=1,verbose=1,scoring='r2')
grs.fit(X,y)
print('Fitting time {} min'.format(round((time()-t)/60,2)))
grs.best_score_
test_df = pd.read_csv('../input/test.csv')
samp = pd.read_csv('../input/sample_submission.csv')
samp.head()
t = time()
pred = list(grs.predict(test_df[['assists', 'boosts', 'damageDealt', 'DBNOs',
       'headshotKills', 'heals', 'killPlace', 'killPoints', 'kills',
       'killStreaks', 'longestKill', 'maxPlace', 'numGroups', 'revives',
       'rideDistance', 'roadKills', 'swimDistance', 'teamKills',
       'vehicleDestroys', 'walkDistance', 'weaponsAcquired', 'winPoints']]))
print('Predicting time {} min'.format(round((time()-t)/60,2)))
len(pred), test_df.shape[0]
ids = list(test_df['Id'])
res  = []
for i in range(len(pred)):
    res.append([ids[i],pred[i]])
out = pd.DataFrame(res,columns=['Id','winPlacePerc'])
out.to_csv('sample1.csv',index=False)
