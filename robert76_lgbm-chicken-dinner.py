import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error , mean_squared_error
import warnings 
warnings.filterwarnings('ignore')
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
columns=['assists', 'boosts', 'damageDealt', 'DBNOs', 'headshotKills', 
                'heals', 'killPlace', 'killPoints', 'kills', 'killStreaks', 
                'longestKill', 'maxPlace', 'numGroups', 'revives','rideDistance', 
                'roadKills', 'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance', 
                'weaponsAcquired', 'winPoints']

X = train.drop(['winPlacePerc','Id'],axis=1)
Y = train.winPlacePerc
#x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size = 0.2, random_state = 42)

#sc = StandardScaler()
#x_train = sc.fit_transform(x_train)
#x_val = sc.transform(x_val)
def runLGBM(train_X, train_y, test_X, seed_val=42):
    params = {
        'boosting_type': 'gbdt', 'objective': 'regression', 'nthread': -1, 'verbose': 0,
        'num_leaves': 31, 'learning_rate': 0.05, 'max_depth': -1,
        'subsample': 0.8, 'subsample_freq': 1, 'colsample_bytree': 0.6, 
        'reg_alpha': 1, 'reg_lambda': 0.001, 'metric': 'rmse',
        'min_split_gain': 0.5, 'min_child_weight': 1, 'min_child_samples': 10, 'scale_pos_weight': 1}
    
    #kf = KFold(n_splits=5, shuffle=True, random_state=seed_val)
    pred_test_y = np.zeros(test_X.shape[0])
    
    train_set = lgb.Dataset(train_X, train_y, silent=True)
        
    model = lgb.train(params, train_set=train_set, num_boost_round=300,verbose_eval=1000)
    pred_test_y = model.predict(test_X, num_iteration = model.best_iteration)
        
    return pred_test_y , model
d_train = lgb.Dataset(x_train, label=y_train)
d_val =lgb.Dataset(x_val,label =y_val)
#params = {"objective" : "regression", "metric" : "mae", 'n_estimators':11000, 'early_stopping_rounds':100,
#              "num_leaves" : 30, "learning_rate" : 0.1, "bagging_fraction" : 0.9,
#               "bagging_seed" : 0}
    
#lgb_model = lgb.train(params, d_train, valid_sets=[d_train , d_val], verbose_eval=1000) 
x_test = test.drop(['Id'],axis=1)
#x_test = sc.fit_transform(x_test)
predicts,model = runLGBM(X, Y, x_test)
sub =pd.DataFrame()
sub['Id']=test.Id
sub['winPlacePerc']=predicts
sub.to_csv('PUBG_LGB.csv',index=False)
