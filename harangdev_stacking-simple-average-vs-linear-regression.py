import numpy as np

import pandas as pd

import gc

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, Ridge, Lasso

from sklearn.model_selection import KFold

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor

from catboost import CatBoostRegressor

import warnings

warnings.filterwarnings("ignore")
data = pd.read_csv('../input/train.csv', index_col='id')

test = pd.read_csv('../input/test.csv', index_col='id')
def process_datetime(df):

    df['date'] = pd.to_datetime(df['date'].astype('str').str[:8])

    df['year'] = df['date'].dt.year

    df['month'] = df['date'].dt.month

    df['day'] = df['date'].dt.day

    df = df.drop('date', axis=1)

    return df



data = process_datetime(data)

test = process_datetime(test)
x_data = data[[col for col in data.columns if col != 'price']]

y_data = np.log1p(data['price'])

del data; gc.collect();
def rmse(pred, true):

    return -np.sqrt(np.mean((pred-true)**2))
max_iterations = 10**5

lgb_model = LGBMRegressor(objective='regression', num_iterations=max_iterations)

xgb_model = XGBRegressor(objective='reg:linear', n_estimators=max_iterations)

cb_model = CatBoostRegressor(loss_function='RMSE', iterations=max_iterations, allow_writing_files = False, depth=4, l2_leaf_reg=1, bootstrap_type='Bernoulli', subsample=0.5)
%%time

archive = pd.DataFrame(columns=['models', 'prediction', 'score'])

for model in [lgb_model, xgb_model, cb_model]:

    models = []

    prediction = np.zeros(len(x_data))

    for t, v in KFold(5, random_state=0).split(x_data):

        x_train = x_data.iloc[t]

        x_val = x_data.iloc[v]

        y_train = y_data.iloc[t]

        y_val = y_data.iloc[v]

        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False, early_stopping_rounds=100)

        models.append(model)

        prediction[v] = model.predict(x_val)

    score = rmse(np.expm1(prediction), np.expm1(y_data))

    print(score)

    archive = archive.append({'models': models, 'prediction': prediction, 'score': score}, ignore_index=True)
test_predictions = np.array([np.mean([model.predict(test) for model in models], axis=0) for models in archive['models']])
archive.head()
mean_stacked_prediction = np.mean([np.expm1(pred) for pred in archive['prediction']], axis=0)

mean_stacked_score = rmse(mean_stacked_prediction, np.expm1(y_data))

mean_stacked_score
x_stack = np.array([np.expm1(pred) for pred in archive['prediction']]).transpose()

y_stack = np.expm1(y_data)
lr_stacker = LinearRegression()

ridge_stacker = RidgeCV(alphas=np.logspace(-2, 3))

lasso_stacker = LassoCV()
%%time

stack_archive = pd.DataFrame(columns=['models', 'prediction', 'score'])

for stacker in [lr_stacker, ridge_stacker, lasso_stacker]:

    prediction = np.zeros(len(x_stack))

    models = []

    for t, v in KFold(5, random_state=0).split(x_stack):

        x_train = x_stack[t]

        x_val = x_stack[v]

        y_train = y_stack.iloc[t]

        y_val = y_stack.iloc[v]

        stacker.fit(x_train, y_train)

        prediction[v] = stacker.predict(x_val)

        models.append(stacker)

    score = rmse(prediction, y_stack)

    print(score)

    stack_archive = stack_archive.append({'models': models, 'prediction': prediction, 'score': score}, ignore_index=True)
stack_archive.head()
# linear regression

np.mean([model.coef_ for model in stack_archive.iloc[0, 0]], axis=0)
# ridge regression

np.mean([model.coef_ for model in stack_archive.iloc[1, 0]], axis=0)
# lasso regression

np.mean([model.coef_ for model in stack_archive.iloc[2, 0]], axis=0)
mean_test_prediction = np.mean(np.expm1(test_predictions), axis=0)
lr_stacker = LinearRegression()

lr_stacker.fit(x_stack, y_stack)

lr_test_prediction = lr_stacker.predict(np.expm1(test_predictions).transpose())
ridge_stacker = Ridge(alpha=np.mean([model.alpha_ for model in stack_archive.iloc[1, 0]]))

ridge_stacker.fit(x_stack, y_stack)

ridge_test_prediction = ridge_stacker.predict(np.expm1(test_predictions).transpose())
lasso_stacker = Lasso(alpha=np.mean([model.alpha_ for model in stack_archive.iloc[2, 0]]))

lasso_stacker.fit(x_stack, y_stack)

lasso_test_prediction = lasso_stacker.predict(np.expm1(test_predictions).transpose())
pd.DataFrame({'id': test.index, 'price': mean_test_prediction}).to_csv('mean_test_prediction.csv', index=False)

pd.DataFrame({'id': test.index, 'price': lr_test_prediction}).to_csv('lr_test_prediction.csv', index=False)

pd.DataFrame({'id': test.index, 'price': ridge_test_prediction}).to_csv('ridge_test_prediction.csv', index=False)

pd.DataFrame({'id': test.index, 'price': lasso_test_prediction}).to_csv('lasso_test_prediction.csv', index=False)