import numpy as np

import pandas as pd

import lightgbm as lgb

import matplotlib.pyplot as plt

import xgboost as xgb

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder



print('Loading data ...')

df_train = pd.read_csv("../input/home-data-for-ml-course/train.csv", index_col="Id")

df_test = pd.read_csv("../input/home-data-for-ml-course/test.csv", index_col="Id")



y = df_train["SalePrice"]

X = df_train.drop(["SalePrice"], axis=1)



#categorical_cols = [c for c in X.columns if X[c].nunique() < 10 and X[c].dtype == "object"]

categorical_cols = [c for c in X.columns if X[c].dtype == "object"]

numerical_cols = [c for c in X.columns if X[c].dtype in [np.int64, np.float64]]

cols = categorical_cols + numerical_cols

#print(cols)



X = X[cols]

X_test = df_test[cols]



numerical_transformer = SimpleImputer(strategy='constant')



categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])



#estimator = lgb.LGBMRegressor()



#param_grid = {

#    'learning_rate': [0.1],

#    'n_estimators': [60, 64],

#    'num_leaves': [31, 33],

#    'reg_alpha': [3],

#    'reg_lambda': [0.1]

#}



#gbm = GridSearchCV(estimator, param_grid, cv=5)



#gbm = lgb.LGBMRegressor(random_state=31, learning_rate=0.1, n_estimators=64, num_leaves=33, reg_alpha=3, reg_lambda=0.1) # 16486.976957731193

gbm = lgb.LGBMRegressor(random_state=31, learning_rate=0.1, n_estimators=60, num_leaves=31, reg_alpha=3, reg_lambda=0.1) # 16457.82515741921



#gbm = xgb.XGBRegressor(random_state=31, learning_rate=0.1, n_estimators=60, num_leaves=31, reg_alpha=3, reg_lambda=0.1)



pipeline = Pipeline(steps=[

    ("preprocessor", preprocessor),

    ("model", gbm)

])



print('Evaluating cross-validation score ...')

scores = -1 * cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')

print("Average MAE score:", scores.mean())



print('Starting training...')

pipeline.fit(X, y)

#print('Best parameters found by grid search are:', gbm.best_params_)

# {'learning_rate': 0.1, 'n_estimators': 64, 'num_leaves': 33, 'reg_alpha': 3, 'reg_lambda': 0.1}



attrs = {k: v for k, v in zip(X.columns, gbm.feature_importances_) if v>0}

attrs = sorted(attrs.items(), key=lambda x: x[1], reverse = False)

x1,y1 = zip(*attrs)

i1=range(len(x1))

plt.figure(figsize=(17, 17), dpi=300, facecolor='w', edgecolor='k')

plt.barh(i1, y1)

plt.title("LGBM")

plt.yticks(i1, x1)

plt.show();



print('Starting predicting ...')

preds_test = pipeline.predict(X_test)



print('Writing results ...')

output = pd.DataFrame({'Id': X_test.index, 'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)