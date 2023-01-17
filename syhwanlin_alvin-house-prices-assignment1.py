import numpy as np
import pandas as pd
import os
print(os.listdir("../input"))
import seaborn as sns
sns.set()
from scipy.stats import skew
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
test.head()
train_Y = train.loc[:, 'SalePrice']

Data = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'], test.loc[:, 'MSSubClass':'SaleCondition']))
sns.distplot(train_Y)
### Right Skewed
sns.distplot(np.log1p(train_Y))
numeric_feats = Data.dtypes[Data.dtypes != "object"].index
skewed_feats = Data[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[(skewed_feats) > 0.75]
skewed_feats = skewed_feats.index
Data[skewed_feats] = np.log1p(Data[skewed_feats])
categorical_feats = Data.dtypes[Data.dtypes == "object"].index
all_data_na = (Data.isnull().sum() / len(Data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
all_data_na[:10]
### One-hot encoding and Imputing
Data = pd.get_dummies(Data)
### Impute Missing value 
Data = Data.fillna(Data.mean())
from sklearn.model_selection import train_test_split
random_seed = 12345

Data_train = Data[:train.shape[0]]
Data_test = Data[train.shape[0]:]
Data_train_Y = np.log1p(train_Y)
train_X, valid_X, train_Y, valid_Y =train_test_split(Data_train, Data_train_Y, test_size=.2, random_state=random_seed)

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
lr_l2_model = Ridge(alpha=10)
lr_l2_model.fit(train_X, train_Y)
from sklearn.metrics import mean_squared_error
lr_pred = lr_l2_model.predict(valid_X)
np.sqrt(mean_squared_error(valid_Y, lr_pred))
### 0.13629545199433987
sns.regplot(x=np.expm1(valid_Y), y=np.expm1(lr_pred), color="b", scatter_kws={'s':4},fit_reg=False)
lr_pred_test = pd.DataFrame(test.loc[:,'Id'])
lr_pred_test.loc[:,'SalePrice'] = np.expm1(lr_l2_model.predict(Data_test))
lr_pred_test.to_csv('lr_pred_test.csv', index=False)
### Score 0.12305
import xgboost as xgb
# model_xgb = xgb.XGBRegressor(n_estimators=250, max_depth=2, learning_rate=0.1)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model_xgb.fit(train_X, train_Y)
xgb_preds = model_xgb.predict(valid_X)
np.sqrt(mean_squared_error(valid_Y, xgb_preds))
### 0.136960132419252355
sns.regplot(x=np.expm1(valid_Y), y=np.expm1(xgb_preds), color="b", scatter_kws={'s':4},fit_reg=False)
xgb_pred_test = pd.DataFrame(test.loc[:,'Id'])
xgb_pred_test.loc[:,'SalePrice'] = np.expm1(model_xgb.predict(Data_test))
xgb_pred_test.to_csv('xgb_pred_test.csv', index=False)
### score 0.12956
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from keras import regularizers
from keras.callbacks import EarlyStopping
Data_nn = StandardScaler().fit_transform(Data)
Data_train_nn = Data_nn[:train.shape[0]]
Data_test_nn = Data_nn[train.shape[0]:]
train_X, valid_X, train_Y, valid_Y =train_test_split(Data_train_nn, Data_train_Y, test_size=.2, random_state=random_seed)
from keras import backend as K
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 
nn_model = Sequential()
nn_model.add(Dense(1, input_shape=(train_X.shape[1],), kernel_regularizer=regularizers.l2(0.01)))
nn_model.compile(loss = "mse", optimizer = "adam" )
nn_model.summary()
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5)
hist = nn_model.fit(train_X, train_Y, validation_data = (valid_X, valid_Y),  epochs=1000)
nn_preds = nn_model.predict(valid_X)
np.sqrt(mean_squared_error(valid_Y, nn_preds))
sns.regplot(x=np.expm1(valid_Y), y=np.expm1(nn_preds), color="b", scatter_kws={'s':4},fit_reg=False)
### Predict Test data
nn_pred_test = pd.DataFrame(test.loc[:,'Id'])
nn_pred_test.loc[:,'SalePrice'] = np.expm1(nn_model.predict(Data_test_nn))
nn_pred_test.to_csv('nn_pred_test.csv', index=False)
### Score 0.13454