import pandas as pd
import numpy as np

df = pd.read_csv( '../input/train_V2.csv', index_col=0 )
df.dropna( inplace=True )
drop_labels = [ 'matchId', 'groupId', 'killPoints', 'roadKills',
                'vehicleDestroys', 'damageDealt', 'rankPoints', 'swimDistance',
                'teamKills', 'winPoints' ]


def train_test_val_split( x, y, test_samples=30_000, val_samples=30_000 ):

    size = len( df )
    
    test_limit = size - test_samples - val_samples
    val_limit = size - val_samples
    
    dtrain, ltrain = x[ : test_limit ], y[ : test_limit ]
    dtest, ltest  = x[ test_limit : val_limit ], y[ test_limit : val_limit ]
    dval, lval   = x[ val_limit : ], y[ val_limit : ]
    
    return ( dtrain, ltrain ), ( dtest, ltest ), ( dval, lval )

dtrain, ltrain = df.drop( labels=drop_labels + ['winPlacePerc'], axis=1 ), df.pop( 'winPlacePerc' )

(dtrain, ltrain), _, (dval, lval) = train_test_val_split( dtrain, ltrain, test_samples=0 )
cfi = [ dtrain.columns.tolist().index( 'matchType' ) ]
f = lambda x: 9 if x > 8 else x

dtrain.headshotKills = dtrain.headshotKills.apply( f )
dval.headshotKills = dval.headshotKills.apply( f )
from catboost import CatBoostRegressor

model = CatBoostRegressor(
    n_estimators = 200,
    loss_function = 'MAE',
    eval_metric = 'RMSE',
    cat_features = cfi )
model.fit( dtrain, ltrain, use_best_model=True, eval_set=(dval, lval), silent=True, plot=True )
df = pd.read_csv( '../input/test_V2.csv', index_col=0 )
df.headshotKills = df.headshotKills.apply( f )
predictions = model.predict( df.drop( labels=drop_labels, axis=1 ) )
import seaborn as sns

sns.distplot( model.predict( dval ) )
from sklearn.metrics import mean_squared_error as mse

mse( model.predict( dval ), lval )
pd.Series( predictions ).describe()
pd.Series( ltrain ).describe()
predictions = np.maximum( predictions, 0 )
predictions = np.minimum( predictions, 1 )
df = pd.read_csv( '../input/sample_submission_V2.csv', index_col=0 )
df.winPlacePerc = predictions
df.to_csv( 'submission.csv' )
for k, v in zip( dtrain.columns, model.feature_importances_ ):
    print( k, '=', v )





