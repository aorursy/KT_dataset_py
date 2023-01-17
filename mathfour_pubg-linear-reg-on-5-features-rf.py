import pandas as pd
import numpy as np
pd.options.display.max_columns = 60
pd.options.display.max_rows = 30
pd.options.display.float_format = lambda x: f' {x:,.2f}'
import warnings
warnings.filterwarnings("ignore")
game = pd.read_csv('../input/train_V2.csv', index_col='Id')
# game = pd.read_csv('input/train_V2.csv', index_col='Id')
game.head()
game.describe()
game.info()
game.isna().sum()
filt = game['winPlacePerc'].isna()
game[filt]
filt2 = game['maxPlace'] == 1
game[filt2]
game = game.fillna(0)
game.isna().sum().sum()
game.corr().style.format("{:.2%}").highlight_min()
X = game[['killPlace','weaponsAcquired','walkDistance','boosts','heals']].values
X[:10]
y = game['winPlacePerc'].values
y[:10]
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
from sklearn.model_selection import cross_val_score
cvs_rfr = cross_val_score(rfr, X, y)
cvs_rfr.mean(), cvs_rfr.std()
rfr.fit(X,y)
game_test = pd.read_csv('../input/test_V2.csv', index_col='Id')
# game_test = pd.read_csv('input/test_V2.csv', index_col='Id')
game_test.head()
game_test.isna().sum().sum()
X_test = game_test[['killPlace','weaponsAcquired','walkDistance','boosts','heals']].values
X_test[:10]
predictions = rfr.predict(X_test).reshape(-1,1)
dfpredictions = pd.DataFrame(predictions, index=game_test.index).rename(columns={0:'winPlacePerc'})
dfpredictions.head(15)
dfpredictions.to_csv('submission.csv', header=True)