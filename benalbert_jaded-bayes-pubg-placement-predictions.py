# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MaxAbsScaler 
from sklearn.metrics import mean_absolute_error
# import csv
test_df = pd.read_csv('../input/train_V2.csv')
# pre-processing and cleaning
test_df = test_df[test_df['winPlacePerc'].notnull()]

cols1 = ['killPlace', 'walkDistance',
         'boosts', 'weaponsAcquired',
         'damageDealt', 'kills',
         'heals', 'longestKill', 'killStreaks', 'assists']
cols2 = ['assists', 'boosts', 'damageDealt',
         'DBNOs', 'headshotKills', 'heals', 'killPlace', 'killPoints',
         'kills', 'killStreaks', 'longestKill', 'matchDuration',
         'maxPlace', 'numGroups', 'rankPoints', 'revives',
         'rideDistance', 'roadKills', 'swimDistance', 'teamKills',
         'vehicleDestroys', 'walkDistance', 'weaponsAcquired', 'winPoints']
X = np.asarray(test_df[cols2], dtype=np.float64)

y = np.asarray(test_df['winPlacePerc'], dtype=np.float64)

# 4 layer network, pre-processing with a StandardScaler

# train test split 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

# scaling
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# fitting model

mlp = MLPRegressor(hidden_layer_sizes=(20, 15, 10, 5))
mod = mlp.fit(X, y)
# predictions = mlp.predict(X_test)

# error = mean_absolute_error(predictions, y_test)
print(mod)
# create submission
final_test = pd.read_csv('../input/test_V2.csv')
# final_test = final_test[final_test['winPlacePerc'].notnull()]

cols = ['assists', 'boosts', 'damageDealt',
         'DBNOs', 'headshotKills', 'heals', 'killPlace', 'killPoints',
         'kills', 'killStreaks', 'longestKill', 'matchDuration',
         'maxPlace', 'numGroups', 'rankPoints', 'revives',
         'rideDistance', 'roadKills', 'swimDistance', 'teamKills',
         'vehicleDestroys', 'walkDistance', 'weaponsAcquired', 'winPoints']
X = np.asarray(final_test[cols], dtype=np.float64)

X = scaler.transform(X)
final_answer = mlp.predict(X)

df = pd.DataFrame({
    'Id': list(final_test['Id']),
    'winPlacePerc': list(final_answer)
})
df.to_csv('sample_submission.csv', index=False)