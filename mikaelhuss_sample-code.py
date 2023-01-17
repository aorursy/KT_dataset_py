# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from hyperopt import fmin, hp, tpe

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split, GridSearchCV, KFold

from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeRegressor



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
labeled_df = pd.read_csv('/kaggle/input/hmboost/train.csv', index_col=0)

test_df = pd.read_csv('/kaggle/input/hmboost/test.csv', index_col=0)
labeled_df.head()
X_train, X_val, y_train, y_val = train_test_split(labeled_df.drop('MEDV', axis=1), labeled_df['MEDV'], test_size=0.3, random_state=42)
decision_tree_model = DecisionTreeRegressor(random_state=2323)

decision_tree_model.fit(X_train, y_train)
pred_val = decision_tree_model.predict(X_val)
rmse = np.sqrt(mean_squared_error(pred_val, y_val))

rmse
pred_val_allmeans = [y_train.mean()]*len(y_val)
rmse_allmeans = np.sqrt(mean_squared_error(pred_val_allmeans, y_val))

rmse_allmeans
test_predictions = decision_tree_model.predict(test_df)

test_predictions
sample_sub = pd.read_csv('/kaggle/input/hmboost/sampleSubmission.csv')

sample_sub.head()
submission_df = pd.DataFrame({'target':test_predictions}).reset_index()

submission_df
submission_df.to_csv('dec_tree.csv', index=None)
dtr = DecisionTreeRegressor(random_state=77)



pipe = Pipeline(steps=[('dtr', dtr)])



# Parameters of pipelines can be set using ‘__’ separated parameter names:

param_grid = {

    'dtr__max_depth': [i+1 for i in range(13)],

    'dtr__max_features': [i+1 for i in range(13)],

}

search = GridSearchCV(pipe, param_grid, n_jobs=-1, scoring='neg_root_mean_squared_error', cv=KFold(3, random_state=88))

search.fit(X_train, y_train)

print("Best parameter (CV score=%0.3f):" % -search.best_score_)

print(search.best_params_)
search.best_params_
grid_tree_model = DecisionTreeRegressor(max_depth=search.best_params_['dtr__max_depth'],

                                        max_features=search.best_params_['dtr__max_features']

                                       )

grid_tree_model.fit(X_train, y_train)

pred_val_grid = grid_tree_model.predict(X_val)

np.sqrt(mean_squared_error(pred_val_grid, y_val))
def evaluate(model, X_train, y_train):

    X_fit, X_eval, y_fit, y_eval = train_test_split(X_train, y_train, test_size=0.3) 

    model.fit(X_fit, y_fit)

    preds = model.predict(X_eval)

    rmse = np.sqrt(mean_squared_error(preds, y_eval))

    return rmse
def opt_fn(params):

    model = DecisionTreeRegressor(max_depth=int(params['max_depth']),

                                  max_features=int(params['max_features']),

                                  random_state=42

                                 )

    rmse = evaluate(model, X_train, y_train)

    return rmse
search_space = {'max_depth': hp.quniform('max_depth', 1, 13, 1),

                'max_features': hp.quniform('max_features', 1, 13, 1),

               }
argmin = fmin(

   fn=opt_fn,

   space=search_space,

   algo=tpe.suggest,

   max_evals=100

)
argmin
hopt_tree_model = DecisionTreeRegressor(max_depth=int(argmin['max_depth']),

                                        max_features=int(argmin['max_features']),

                                        random_state=23

                                       )

hopt_tree_model.fit(X_train, y_train)

pred_val_hopt = hopt_tree_model.predict(X_val)

np.sqrt(mean_squared_error(pred_val_hopt, y_val))
import xgboost



xg = xgboost.XGBRegressor()

xg.fit(X_train, y_train)

pred_val_xg = xg.predict(X_val)



# If you want to use early stopping

# Here you may want to split the training data again

# instead of using X_val to avoid overfitting the the validation set

xg.fit(X_train,

       y_train,

       early_stopping_rounds=50,

       eval_metric="rmse",

       eval_set=[(X_val, y_val)])
import lightgbm as lgb



parameters = {

    'boosting': 'gbdt',

    'metric': 'mse',

    'verbose': 0

}



lgb_train = lgb.Dataset(X_train, y_train)

lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)



light = lgb.train(parameters,

                  lgb_train)



pred_val_light = light.predict(X_val)



# If you want to use early stopping

# Here you may want to split the training data again

# instead of using lgb_eval to avoid overfitting the the validation set

light_es = lgb.train(parameters,

                     lgb_train,

                     valid_sets=lgb_eval,

                     early_stopping_rounds=5)