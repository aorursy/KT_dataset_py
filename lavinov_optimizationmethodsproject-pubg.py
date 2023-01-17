import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from hyperopt import hp, fmin, STATUS_OK, tpe

dataset = pd.read_csv('../input/train_V2.csv')
dataset
dataset.shape
dataset.describe()
dataset.columns[dataset.isna().any()].tolist()
pd.set_option('display.max_columns', 30)
null_columns=dataset.columns[dataset.isnull().any()]
print(dataset[dataset["winPlacePerc"].isnull()][null_columns])
dataset = dataset.drop(dataset.index[2744604])
dataset.columns[dataset.isna().any()].tolist()
dataset.corr()
subdataset = dataset.sample(n=300000, random_state=4241)

dataset_y = subdataset['winPlacePerc']
dataset_X = subdataset.drop(['Id', 'groupId', 'matchId', 'winPlacePerc'], axis=1)

dataset_X = pd.get_dummies(dataset_X)
dataset_X.shape
# dataset_X_without_features = dataset_X.drop(['vehicleDestroys', 'teamKills', 'roadKills', 'rankPoints', 'killPoints'], axis=1)

# X_train1, X_test1, y_train1, y_test1 = train_test_split(dataset_X_without_features, dataset_y, test_size=0.33, random_state=42)

# model = xgb.XGBRegressor()
# print('>>> Start learning')
# model.fit(X_train1, y_train1)
# print('>>> Learning finished, starting prediction')

# predicted_y1 = model.predict(X_test1)

# print('>>> Prediction finished, counting MAE')

# mae = mean_absolute_error(y_test1, predicted_y1)


# print(mae)
# Все же попробую гипероптимизировать со всеми признаками
X_train1, X_test1, y_train1, y_test1 = train_test_split(dataset_X, dataset_y, test_size=0.33, random_state=42)
# def score(params):
#     print('Learning with params:')
#     print(params)
    
#     model = xgb.XGBRegressor(**params)
#     model.fit(X_train1, y_train1)
    
#     predictions = model.predict(X_test1)
#     score = mean_absolute_error(y_test1, predictions)
        
#     print("\tScore {0}\n\n".format(score))
#     loss = score
    
#     return { 'loss': loss, 'status': STATUS_OK }
# def optimize():
#     space = {
#         'n_estimators': hp.choice('n_estimators', np.arange(1, 1000, 1, dtype=int)),
#         'learning_rate': hp.quniform('learning_rate', 0.025, 0.5, 0.025),
        
#         'max_depth': hp.choice('max_depth', np.arange(6, 8, dtype=int)),
#         'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
#         'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
#         'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
#         'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
#         'silent': 1,
#         'random_state': 42
#     }
    
#     best = fmin(score, space, algo=tpe.suggest, max_evals=250)
#     return best
# best_hyperparams = optimize()

# print("The best hyperparameters are: ", "\n")
# print(best_hyperparams)
# model = xgb.XGBRegressor(colsample_bytree=0.75, 
#                          gamma=0.7000000000000001, 
#                          learning_rate=0.25, 
#                          max_depth=6,
#                          min_child_weight=3.0,
#                          n_estimators=690,
#                          random_state=42, 
#                          silent=1,
#                          subsample=0.75)
# model.fit(X_train1, y_train1)
    
# predictions = model.predict(X_test1)

# score = mean_absolute_error(y_test1, predictions)

# print(score)
model = xgb.XGBRegressor(colsample_bytree=0.9, 
                         gamma=0.55, 
                         learning_rate=0.07500000000000001, 
                         max_depth=7,
                         min_child_weight=5.0,
                         n_estimators=830,
                         random_state=42, 
                         silent=1,
                         subsample=0.75)
model.fit(X_train1, y_train1)
    
predictions = model.predict(X_test1)

score = mean_absolute_error(y_test1, predictions)

print(score)
subdataset2 = dataset.sample(n=300000, random_state=8425)

dataset_y_2 = subdataset2['winPlacePerc']
dataset_X_2 = subdataset2.drop(['Id', 'groupId', 'matchId', 'winPlacePerc'], axis=1)

dataset_X_2 = pd.get_dummies(dataset_X_2)

X_train2, X_test2, y_train2, y_test2 = train_test_split(dataset_X_2, dataset_y_2, test_size=0.33, random_state=42)
model2 = xgb.XGBRegressor(colsample_bytree=0.9, 
                         gamma=0.55, 
                         learning_rate=0.07500000000000001, 
                         max_depth=7,
                         min_child_weight=5.0,
                         n_estimators=830,
                         random_state=42, 
                         silent=1,
                         subsample=0.75)
model2.fit(X_train2, y_train2)
    
predictions2 = model2.predict(X_test2)

score2 = mean_absolute_error(y_test2, predictions2)

print(score2)
subdataset3 = dataset.sample(n=300000, random_state=98530)

dataset_y_3 = subdataset3['winPlacePerc']
dataset_X_3 = subdataset3.drop(['Id', 'groupId', 'matchId', 'winPlacePerc'], axis=1)

dataset_X_3 = pd.get_dummies(dataset_X_3)

X_train3, X_test3, y_train3, y_test3 = train_test_split(dataset_X_3, dataset_y_3, test_size=0.33, random_state=42)
model3 = xgb.XGBRegressor(colsample_bytree=0.9, 
                         gamma=0.55, 
                         learning_rate=0.07500000000000001, 
                         max_depth=7,
                         min_child_weight=5.0,
                         n_estimators=830,
                         random_state=42, 
                         silent=1,
                         subsample=0.75)
model3.fit(X_train3, y_train3)
    
predictions3 = model3.predict(X_test3)

score3 = mean_absolute_error(y_test3, predictions3)

print(score3)
origin_test_dataset = pd.read_csv('../input/test_V2.csv')
test_dataset = origin_test_dataset.drop(['Id', 'groupId', 'matchId'], axis=1)
test_dataset
test_dataset = pd.get_dummies(test_dataset)
test_dataset.shape
test_predictions1 = model.predict(test_dataset)
test_predictions2 = model2.predict(test_dataset)
test_predictions3 = model3.predict(test_dataset)
test_predictions1
test_predictions2
test_predictions3
test_predictions_mean = (test_predictions1 + test_predictions2 + test_predictions3) / 3
test_predictions_mean
submit_df = test_dataset[:]
submit_df['winPlacePerc'] = test_predictions_mean
submit_df['Id'] = origin_test_dataset['Id']
submit_df
submit_df = submit_df.loc[:, ['Id', 'winPlacePerc']]
submit_df.columns
submit_df
submit_df.to_csv('submit3.csv', index=False)
