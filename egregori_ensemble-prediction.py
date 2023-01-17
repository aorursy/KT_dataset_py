import numpy as np
import pandas as pd
import os

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
np.seterr(divide='ignore', invalid='ignore')

print(os.listdir('../input'))
df = pd.read_csv('../input/train.csv')
target_column = 'SalePrice'
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric_df = df.select_dtypes(include=numerics)
numeric_df.columns
observed_column = 'TotalBsmtSF'
# df[observed_column] = df['YrSold'] + (df['MoSold'] * 0.1)
df[observed_column].hist(bins=50)
print(df[observed_column].describe())
df.plot(x=observed_column, y=target_column, style='o')
non_numeric_df = df.select_dtypes(include=['object'])
non_numeric_df.columns
df.groupby('MasVnrType').count()
df['GarageFinish'].head()
non_numeric_df = df.select_dtypes(include=['object'])
non_numeric_df = non_numeric_df.join(df[target_column])
deleted_column = ['Alley', 'Utilities', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                 'BsmtFinType2', 'Electrical', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 
                  'GarageCond', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
for col in deleted_column:
    del non_numeric_df[col]

threshold = 0.8
one_hot_columns = []
for column in non_numeric_df.columns:
    if column != target_column:
        X = df[target_column].values
        y = df[column].values
        X = X.reshape(len(X), 1)
        x_t, x_v, y_t, y_v = train_test_split(X, y, test_size=0.2, random_state=77)
        lr = LogisticRegression().fit(x_t, y_t)
        if accuracy_score(y_v, lr.predict(x_v)) > threshold:
            one_hot_columns.append(column)
def make_one_hot(df, columns):
    for column in columns:
        temp = pd.get_dummies(df[column], prefix=column)
        df = df.join(temp)
        
    return df
fillna_with_zero = ['TotalBsmtSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'YearBuilt',
                   'LowQualFinSF', 'TotRmsAbvGrd', 'GarageYrBlt']
fillna_with_median = ['OverallQual', '1stFlrSF']
fillna_with_max = ['LotFrontage']
def preprocess(df, train_df):
    df[fillna_with_zero] = df[fillna_with_zero].fillna(0)
    df[fillna_with_median] = df[fillna_with_median].fillna(train_df[fillna_with_median].median())
    df[fillna_with_max] = df[fillna_with_max].fillna(train_df[fillna_with_median].max())
    
    df['far_from_street'] = np.where(df['LotFrontage'] > 70, 1, 0)
    df['new_house'] = np.where(df['YearBuilt'] > 1990, 1, 0)
    df['has_second_floor'] = np.where(df['2ndFlrSF'] > 0, 1, 0)
    df['has_unfinished_floor'] = np.where(df['LowQualFinSF'] > 0, 1, 0)
    df['room_above_grade'] = np.where(df['TotRmsAbvGrd'] > 4, 1, 0)
    df['new_garage'] = np.where(df['GarageYrBlt'] > 1990, 1, 0)
    
    df = make_one_hot(df, one_hot_columns)
    
    return df
feature_columns = ['OverallQual', 
                   'TotalBsmtSF', 
                   '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 
                  'WoodDeckSF', 'far_from_street', 'new_house', 
                   'has_second_floor', 
                   'new_garage',
                   'has_unfinished_floor', 
                   'room_above_grade']
for column in one_hot_columns:
    [feature_columns.append(col) for col in list(df.columns) if col.startswith(column + '_')]
df = pd.read_csv('../input/train.csv')
df = preprocess(df, df)
X = df[feature_columns].values
y = df[target_column].values
mins = np.min(X, axis=0)
maxs = np.max(X, axis=0)
rng = maxs - mins
# X = 1 - ((maxs - X) / rng)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import History 
from keras.utils import plot_model
from keras.optimizers import SGD
def crossval_rmse(regr, X, y, cv=10):
    rmse = []
    for state, i in enumerate(range(cv)):
        X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=state)
        temp_model = regr.fit(X_train, y_train)
        prediction = temp_model.predict(X_validation)
        try:
            rmse.append(np.sqrt(mean_squared_error(np.log(y_validation), np.log(prediction))))
        except:
            pass
    print(np.mean(rmse))
lr = LinearRegression()
crossval_rmse(lr, X, y)
lasso = Lasso(alpha =0.1, random_state=1)
crossval_rmse(lasso, X, y)
enet = ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)
crossval_rmse(enet, X, y)
gboost = GradientBoostingRegressor()
crossval_rmse(gboost, X, y)
model_xgb = xgb.XGBRegressor()
crossval_rmse(model_xgb, X, y)
model_lgb = lgb.LGBMRegressor()
crossval_rmse(model_lgb, X, y)
np.random.seed(77)
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=77)
neurons = [2, 2, 2, 1]
initial_weight = np.random.rand(X.shape[1], neurons[0])
initial_bias = np.zeros(neurons[0])
nn_model = Sequential()
nn_model.add(Dense(neurons[0], weights=[initial_weight, initial_bias], activation='relu', input_dim = X.shape[1]))
for n in range(1, len(neurons)):
    nn_model.add(Dense(neurons[n], activation='relu'))
nn_model.compile(optimizer='adam', loss='mean_squared_error')
nn_model.fit(X_train, y_train, epochs=100, validation_split=.1, verbose=0)
prediction = nn_model.predict(X_validation)
print(np.sqrt(mean_squared_error(np.log(y_validation), np.log(prediction))))
def get_models(regrs, X, y):
    models = []
    for regr in regrs:
        models.append(regr.fit(X, y))
    
    return models
def predict_with_ensemble_models(models, x):
    predicts = []
    for model in models:
        predicts.append(model.predict(x))
    
    return np.mean(predicts, axis=0)
def crosvall_ensemble(regrs, X, y, cv=10):
    rmse = []
    for state, i in enumerate(range(cv)):
        X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=state)
        temp_models = get_models(regrs, X_train, y_train)
        prediction = predict_with_ensemble_models(temp_models, X_validation)
        try:
            rmse.append(np.sqrt(mean_squared_error(np.log(y_validation), np.log(prediction))))
        except:
            pass
    print(np.mean(rmse))
crosvall_ensemble([lr, lasso, enet], X, y)
crosvall_ensemble([gboost, model_xgb, model_lgb], X, y)
crosvall_ensemble([lr, lasso, enet, gboost, model_xgb, model_lgb], X, y)
def train_stacking_nn(regrs, X, y, cv=10):
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=77)
    temp_models = get_models(regrs, X_train, y_train)
    ensemble_train = []
    ensemble_validation = []
    for model in temp_models:
        ensemble_train.append(model.predict(X_train))
        ensemble_validation.append(model.predict(X_validation))
    ensemble_train = np.reshape(ensemble_train, (len(y_train), len(regrs)))
    ensemble_validation = np.reshape(ensemble_validation, (len(y_validation), len(regrs)))

    np.random.seed(77)
    neurons = [2, 1]
    initial_weight = np.random.rand(ensemble_train.shape[1], neurons[0])
    initial_bias = np.zeros(neurons[0])
    nn_model = Sequential()
    nn_model.add(Dense(neurons[0], activation='relu', weights=[initial_weight, initial_bias], 
                 input_dim = ensemble_train.shape[1]))
    for n in range(1, len(neurons)):
        nn_model.add(Dense(neurons[n], activation='relu'))
    nn_model.compile(optimizer='adam', loss='mean_squared_error')
    nn_model.fit(ensemble_train, y_train, epochs=100, validation_split=.1, verbose=0)
    prediction = nn_model.predict(ensemble_validation)
    print(np.sqrt(mean_squared_error(np.log(y_validation), np.log(prediction))))
train_stacking_nn([gboost, model_xgb, model_lgb], X, y)
def train_stacking(stacking_regr, regrs, X, y, cv=10):
    rmse = []
    for state, i in enumerate(range(cv)):
        X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=state)
        temp_models = get_models(regrs, X_train, y_train)
        ensemble_train = []
        ensemble_validation = []
        for model in temp_models:
            ensemble_train.append(model.predict(X_train))
            ensemble_validation.append(model.predict(X_validation))
        ensemble_train = np.reshape(ensemble_train, (len(y_train), len(regrs)))
        ensemble_validation = np.reshape(ensemble_validation, (len(y_validation), len(regrs)))
        
        temp_model = stacking_regr.fit(ensemble_train, y_train)
        prediction = temp_model.predict(ensemble_validation)
        try:
            rmse.append(np.sqrt(mean_squared_error(np.log(y_validation), np.log(prediction))))
        except:
            pass
    print(np.mean(rmse))
train_stacking(model_xgb, [lr, lasso, enet, gboost, model_xgb, model_lgb], X, y)
train_stacking(gboost, [lr, lasso, enet], X, y)
train_stacking(gboost, [gboost, model_xgb, model_lgb], X, y)
def train_blending(blending_regr, regrs, X, y, cv=10):
    rmse = []
    for state, i in enumerate(range(cv)):
        X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=state)
        temp_models = get_models(regrs, X_train, y_train)
        ensemble_train = []
        ensemble_validation = []
        for model in temp_models:
            ensemble_train.append(model.predict(X_train))
            ensemble_validation.append(model.predict(X_validation))
        ensemble_train = np.reshape(ensemble_train, (len(y_train), len(regrs)))
        ensemble_validation = np.reshape(ensemble_validation, (len(y_validation), len(regrs)))
        
        X_train = np.append(X_train, ensemble_train, axis=1)
        X_validation = np.append(X_validation, ensemble_validation, axis=1)
        temp_model = blending_regr.fit(X_train, y_train)
        prediction = temp_model.predict(X_validation)
        try:
            rmse.append(np.sqrt(mean_squared_error(np.log(y_validation), np.log(prediction))))
        except:
            pass
    print(np.mean(rmse))
regrs = [lr, lasso, enet, gboost, model_xgb, model_lgb]
for regr in regrs:
    train_blending(regr, regrs, X, y)
regrs = [gboost, model_xgb, model_lgb]
for regr in regrs:
    train_blending(regr, regrs, X, y)
models = get_models([gboost, model_xgb, model_lgb], X, y)
df_test = pd.read_csv('../input/test.csv')
df_test = preprocess(df_test, df)
for col in feature_columns:
    if col not in df_test.columns:
        df_test[col] = 0
prediction = predict_with_ensemble_models(models, df_test[feature_columns].values)
prediction_dict = {
    'Id': df_test['Id'],
    target_column: prediction
}
prediction_df = pd.DataFrame.from_dict(prediction_dict)
prediction_df.to_csv('prediction.csv', index=False)
#TODO add more feature, use antoher ensemble method
