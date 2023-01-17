import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer



from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV



from sklearn.linear_model import LinearRegression, ElasticNet

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor, VotingRegressor



from xgboost import XGBRegressor
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
sample_submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
train.head()
print(train.shape)
test.head()
print(test.shape)
plt.figure(figsize=[12, 4])

plt.subplot(1, 2, 1)

plt.hist(train.SalePrice, bins=20, color='plum', edgecolor='k')



plt.subplot(1, 2, 2)

plt.hist(np.log(train.SalePrice), bins=20, color='plum', edgecolor='k')



plt.show()
X_train = train.drop(['Id', 'SalePrice'], axis=1)

y_train = np.log(train.SalePrice)

X_test = test.drop(['Id'], axis=1)



print('X_train shape: ', X_train.shape)

print('y_triain shape: ', y_train.shape)

print('X_test shape: ', X_test.shape)
X_train.isna().sum().sort_values(ascending=False)[:20]
print(np.unique(X_train.dtypes.values))
sel_num = (X_train.dtypes.values == 'int64') | (X_train.dtypes.values == 'float64')

num_idx = np.arange(0, len(X_train.columns))[sel_num]

X_train_num = X_train.iloc[:, num_idx]



print('Number of Numerical Columns: ', np.sum(sel_num), '\n')

print('Indices for Numerical Columns: ', num_idx, '\n')

print('Namse of Numerical Columns:\n', X_train_num.columns.values)
sel_cat = (X_train.dtypes.values == 'O')

cat_idx = np.arange(0, len(X_train.columns))[sel_cat]

X_train_cat = X_train.iloc[:, cat_idx]



print('Number of Categorical Columns: ', np.sum(sel_cat), '\n')

print('Indices for Categorical Columns: ', cat_idx, '\n')

print('Namse of Categorical Columns:\n', X_train_num.columns.values)
num_transformer = Pipeline(

    steps=[

        ('imputer', SimpleImputer(strategy='mean')),

        #('scaler', StandardScaler())

    ]

)



cat_transformer = Pipeline(

    steps=[

        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),

        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))

    ]

)



preprocessor = ColumnTransformer(

    transformers = [

        ('num', num_transformer, num_idx),

        ('cat', cat_transformer, cat_idx)

    ]

)



preprocessor.fit(X_train)

train_proc = preprocessor.transform(X_train)

print(train_proc.shape, '\n')

encoded_names = preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names(X_train_cat.columns.values)

print(encoded_names[:20])



features_names = np.concatenate([X_train_num.columns.values, encoded_names])

print(len(features_names))
lr_pipe = Pipeline(

    steps = [

        ('preprocessor', preprocessor),

        ('regressor', LinearRegression())

    ]

)

lr_pipe.fit(X_train, y_train)

lr_pipe.score(X_train, y_train)
cv_results = cross_val_score(lr_pipe, X_train, y_train, cv=10, scoring='neg_root_mean_squared_error')

print('Results by fold:\n', cv_results, '\n')

print('Mean CV Score: ', np.mean(cv_results))
%%time

en_pipe = Pipeline(

    steps = [

        ('preprocessor', preprocessor),

        ('regressor', ElasticNet(max_iter=1000))

    ]

)

param_grid = {

    'regressor__alpha':[0.0001, 0.001, 0.01, 0.1],

    'regressor__l1_ratio':[0, 0.25, 0.5, 0.75, 1.0],

}



np.random.seed(1)

en_grid_search = GridSearchCV(en_pipe, param_grid, cv=10, scoring = 'neg_root_mean_squared_error', refit='True', verbose = 10, n_jobs=-1)

en_grid_search.fit(X_train, y_train)



print(en_grid_search.best_score_)

print(en_grid_search.best_params_)
en_model = en_grid_search.best_estimator_.steps[1][1]

print('Number of Features Kept: ', np.sum(en_model.coef_ != 0))

print('Number of Features Dropped: ', np.sum(en_model.coef_ == 0))
cv_results = cross_val_score(en_grid_search.best_estimator_, X_train, y_train, cv=10, scoring= 'r2')

print('Results by fold:\n', cv_results, '\n')

print('Mean CV score:', np.mean(cv_results))
%%time 



dt_pipe = Pipeline(

    steps = [

        ('preprocessor', preprocessor),

        ('regressor', DecisionTreeRegressor())

    ]

)



param_grid = {

    'regressor__min_samples_leaf': [8, 16, 32, 64],

    'regressor__max_depth': [8, 16, 32, 64],

}



np.random.seed(1)

dt_grid_search = GridSearchCV(dt_pipe, param_grid, cv=10, scoring='neg_root_mean_squared_error',

                              refit='True', verbose = 10, n_jobs=-1)

dt_grid_search.fit(X_train, y_train)



print(dt_grid_search.best_score_)

print(dt_grid_search.best_params_)
%%time 



rf_pipe = Pipeline(

    steps = [

        ('preprocessor', preprocessor),

        ('regressor', RandomForestRegressor(n_estimators=100))

    ]

)



param_grid = {

    'regressor__min_samples_leaf': [8, 16, 32],

    'regressor__max_depth': [4, 8, 16, 32],

}



np.random.seed(1)

rf_grid_search = GridSearchCV(rf_pipe, param_grid, cv=10, scoring='neg_root_mean_squared_error',

                              refit='True', verbose = 10, n_jobs=-1)

rf_grid_search.fit(X_train, y_train)



print(rf_grid_search.best_score_)

print(rf_grid_search.best_params_)
rf_model = rf_grid_search.best_estimator_.steps[1][1]
feat_imp = rf_model.feature_importances_

feat_imp_df = pd.DataFrame({

    'feature':feature_names,

    'feat_imp':feat_imp

})



feat_imp_df.sort_values(by='feat_imp', ascending=False).head(10)
feat_imp_df.sort_values(by='feat_imp').head(10)
sorted_feat_imp_df = feat_imp_df.sort_values(by='feat_imp', ascending=True)

plt.figure(figsize=[6,6])

plt.barh(sorted_feat_imp_df.feature[-20:], sorted_feat_imp_df.feat_imp[-20:])

plt.show()
%%time 



xgd_pipe = Pipeline(

    steps = [

        ('preprocessor', preprocessor),

        ('regressor', XGBRegressor(n_estimators=50, subsample=0.5))

    ]

)



param_grid = {

    'regressor__learning_rate' : [0.1, 0.5, 0.9],

    'regressor__alpha' : [0, 1, 10],

    'regressor__max_depth': [4, 8, 16]

    

}



np.random.seed(1)

xgd_grid_search = GridSearchCV(xgd_pipe, param_grid, cv=10, scoring='neg_root_mean_squared_error',

                              refit='True', verbose = 10, n_jobs=-1)

xgd_grid_search.fit(X_train, y_train)



print(xgd_grid_search.best_score_)

print(xgd_grid_search.best_params_)
xgb_model = xgd_grid_search.best_estimator_.steps[1][1]
ensemble = VotingRegressor(

    estimators = [

        ('en', en_grid_search.best_estimator_),

        ('rf', rf_grid_search.best_estimator_),

        ('xgb', xgd_grid_search.best_estimator_),

    ]

)



cv_results = cross_val_score(ensemble, X_train, y_train, cv=10, scoring='neg_root_mean_squared_error')



print('Results by fold:\n', cv_results, '\n')

print('Mean CV Score:', np.mean(cv_results))
cv_results = cross_val_score(ensemble, X_train, y_train, cv=10, scoring='r2')



print('Results by fold:\n', cv_results, '\n')

print('Mean CV Score:', np.mean(cv_results))
ensemble.fit(X_train, y_train)

ensemble.score(X_train, y_train)
sample_submission.head()
submission = sample_submission.copy()

submission.SalePrice = np.exp(ensemble.predict(X_test))



submission.to_csv('my_submission.csv', index=False)

submission.head()