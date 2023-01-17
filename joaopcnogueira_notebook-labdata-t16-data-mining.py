import os

import numpy as np

import pandas as pd
DATA_DIR = '../input/house-prices-advanced-regression-techniques'



train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
train_df.head()
train_df.shape
train_df['Id'].nunique()
train_df.isnull().sum()
target = 'SalePrice'

cat_vars = train_df.select_dtypes(include='object').columns.to_list()

num_vars = [col for col in train_df.columns if col not in cat_vars + ['Id', target]]
X = train_df.filter(cat_vars + num_vars).copy()

y = train_df[target].copy()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=30)
!pip install feature-engine
from feature_engine.missing_data_imputers import ArbitraryNumberImputer, CategoricalVariableImputer

from feature_engine.categorical_encoders import OneHotCategoricalEncoder

from sklearn.pipeline import Pipeline



data_pipe = Pipeline(steps=[

    ('numeric_imputer', ArbitraryNumberImputer(arbitrary_number=-999, variables=num_vars)),

    ('categoric_imputer', CategoricalVariableImputer(fill_value='Missing', variables=cat_vars, return_object=True)),

    ('one_hot_encoder', OneHotCategoricalEncoder(variables=cat_vars))

])
data_pipe.fit_transform(X_train)
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.pipeline import make_pipeline
# linear_regression = make_pipeline(StandardScaler(), LinearRegression())



models = [

#     ('linear_regression', linear_regression),

    ('decision_tree', DecisionTreeRegressor(random_state=50)),

    ('random_forest', RandomForestRegressor(random_state=50)),

    ('lgbm', LGBMRegressor(random_state=50)),

    ('xgb', XGBRegressor(random_state=50))

]
from sklearn.model_selection import cross_val_score



training_results = []

for model in models:



    model_pipe = Pipeline(steps=data_pipe.steps + [model]) # adiciona o modelo no fim do pipeline de dados



    cv_results_r2 = cross_val_score(model_pipe, X_train, y_train, scoring='r2', cv=5, n_jobs=-1)

    cv_results_mae = cross_val_score(model_pipe, X_train, y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)

    cv_results_rmse = cross_val_score(model_pipe, X_train, y_train, scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1)

    cv_results_rmsle = cross_val_score(model_pipe, X_train, y_train, scoring='neg_mean_squared_log_error', cv=5, n_jobs=-1)



    r2 = np.abs(cv_results_r2.mean())

    mae = np.abs(cv_results_mae.mean())

    rmse = np.abs(cv_results_rmse.mean())

    rmsle = np.sqrt(np.abs(cv_results_rmsle.mean()))

    

    training_results.append([model[0], r2, mae, rmse, rmsle])

#     training_results.append([model[0], r2, mae, rmse])





training_results_df = pd.DataFrame(training_results, columns=['model', 'r2', 'mae', 'rmse', 'rmsle'])

# training_results_df = pd.DataFrame(training_results, columns=['model', 'r2', 'mae', 'rmse'])
training_results_df
best_model = Pipeline(data_pipe.steps + [('lgbm', LGBMRegressor(random_state=50))])



best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)



import matplotlib.pyplot as plt

import seaborn as sns



fig, ax = plt.subplots(figsize=(7,5))

sns.regplot(y_test, y_pred, ax=ax);

ax.set_xlabel('Real Values');

ax.set_ylabel('Predictions');
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score



mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)



print(f"MAE = {mae:.3f}")

print(f"RMSE = {rmse:.3f}")

print(f"RMSLE = {rmsle:.3f}")

print(f"R2 = {r2:.3f}")
best_model = Pipeline(data_pipe.steps + [('lgbm', LGBMRegressor(random_state=50))])



best_model.fit(X, y)

y_pred = best_model.predict(X)
fig, ax = plt.subplots(figsize=(7,5))

sns.regplot(y, y_pred, ax=ax);

ax.set_xlabel('Real Values');

ax.set_ylabel('Predictions');
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score



mae = mean_absolute_error(y, y_pred)

mse = mean_squared_error(y, y_pred)

rmse = np.sqrt(mse)

rmsle = np.sqrt(mean_squared_log_error(y, y_pred))

r2 = r2_score(y, y_pred)



print(f"MAE = {mae:.3f}")

print(f"RMSE = {rmse:.3f}")

print(f"RMSLE = {rmsle:.3f}")

print(f"R2 = {r2:.3f}")
oot_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
oot_df.head()
X_oot = oot_df.filter(cat_vars + num_vars).copy()

y_pred_oot = best_model.predict(X_oot)



submission = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

submission['SalePrice'] = y_pred_oot
OUTPUT_DIR = '/kaggle/working'

submission.to_csv(os.path.join(OUTPUT_DIR, 'submission-kaggle-T16-data-mining-v3.csv'), index=False)