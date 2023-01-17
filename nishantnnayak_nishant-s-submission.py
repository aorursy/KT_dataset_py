# Importing the necessary libraries

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error as mae
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
# Importing the data
base_path = "../input/wecrec2020"
train_data = pd.read_csv(base_path + "/Train_data.csv")
test_data = pd.read_csv(base_path + "/Test_data.csv")
# Dropping columns 1 and 2 since they have no effect on output variable
train_data_drop = train_data.drop(['Unnamed: 0', 'F1', 'F2'], axis=1)
test_X = test_data.drop(['Unnamed: 0', 'F1', 'F2'], axis=1)
# Splitting into input and output variables
X = train_data_drop.loc[:, 'F3':'F17']
Y = train_data_drop.loc[:, 'O/P']
objs = pd.concat(objs=[X, test_X])
cat_cols = ['F3','F4','F5','F7','F8','F9','F11','F12']
combo = pd.get_dummies(data=objs, columns=cat_cols, dtype=np.float64, drop_first=True)

X_dummy = pd.DataFrame(data=combo[0:Y.shape[0]])
test_X_dummy = pd.DataFrame(data=combo[Y.shape[0]:])
X_dummy.head()
cont_cols = [col for col in X.columns if col not in cat_cols]
scaler = RobustScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_dummy[cont_cols]))
X_scaled.columns = cont_cols

test_X_scaled = pd.DataFrame(scaler.transform(test_X_dummy[cont_cols]))
test_X_scaled.columns = cont_cols

X_preprocessed = pd.concat([X_scaled, X_dummy.loc[:, 'F3_2':]], axis=1)
test_X_preprocessed = pd.concat([test_X_scaled, test_X_dummy.loc[:, 'F3_2':]], axis=1)
X_preprocessed.head()
train_X, val_X, train_y, val_y = train_test_split(X_preprocessed, Y, random_state=0, test_size=0.1)
my_model = XGBRegressor(random_state=0, n_estimators=300, reg_lambda=7, max_depth=5)
my_model.fit(train_X, train_y)

preds_0 = my_model.predict(val_X)
print(mae(val_y, preds_0))
my_model_2 = XGBRegressor(random_state=0, n_estimators=200, reg_alpha=1, max_depth=5, subsample=0.85)
my_model_2.fit(X_preprocessed, Y)

preds_test = my_model_2.predict(test_X_preprocessed)
preds_test
my_model_3 = lgb.LGBMRegressor(random_state=0, n_estimators=200, reg_lambda=1.0)

my_model_3.fit(train_X, train_y, eval_metric='l1')
pred_3 = my_model_3.predict(val_X)

print(mae(val_y, pred_3))
my_model_4 = RandomForestRegressor(random_state=0)

my_model_4.fit(train_X, train_y)
preds_4 = my_model_4.predict(val_X)

print(mae(val_y, preds_4))
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, PReLU
from tensorflow.keras import regularizers
my_model_nn = Sequential([
    Dense(56, input_shape=[56,], activation="relu"),
    Dense(28, activation="tanh", kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.2),
    Dense(28, activation="relu", kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.2),
    Dense(14, activation="relu", kernel_regularizer=regularizers.l2(0.01)),
    Dense(1)
])
my_model_nn.compile(optimizer='adam', loss='mean_absolute_error')

my_model_nn.fit(train_X, train_y, validation_data=(val_X, val_y), epochs=200)
preds_nn = my_model_nn.predict(val_X)
print(mae(val_y, preds_nn))
my_model_nn.fit(X_preprocessed, Y, epochs=300)
preds_final = my_model_nn.predict(test_X_preprocessed)
preds_final
result = pd.DataFrame()

result['ID'] = test_data['Unnamed: 0']
result['PredictedValue'] = pd.DataFrame(preds_final)
result.head()
result.to_csv('output.csv', index=False)
