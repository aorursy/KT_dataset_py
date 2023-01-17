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
pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', 20)
sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
print("Number of rows and columns in the train dataset are:", train.shape)
train.head()
train.tail()
train.info()
train.describe()
train.isna().sum().sort_values(ascending=False)[:20]
import seaborn as sns

# Let's plot these missing values(%) vs column_names

missing_values_count = (train.isnull().sum()/train.isnull().count()*100).sort_values(ascending=False)

plt.figure(figsize=(15,5))

base_color = sns.color_palette()[0]

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)

sns.barplot(missing_values_count[:10].index.values, missing_values_count[:10], color = base_color)
y_train = np.log(train.SalePrice)
# Features with over 50% of its observations missings will be removed

train = train.drop(['Id','SalePrice','PoolQC','MiscFeature','Alley','Fence'],axis = 1)


X_test = test.drop(['Id'], axis=1)
train.dtypes.value_counts()
sel_num = (train.dtypes.values == 'int64') | (train.dtypes.values == 'float64')

num_idx = np.arange(0, len(train.columns))[sel_num]

train_num = train.iloc[:, num_idx]



print('Number of Numerical Columns:  ', np.sum(sel_num), '\n')

print('Indices for Numerical Columns:', num_idx, '\n')

print('Names of Numerical Columns:\n', train_num.columns.values)
con  = (train_num.dtypes.values == 'float64')

idx=np.arange(0, len(train_num.columns))[con]

train_num[train_num.iloc[:,idx].columns] = train_num[train_num.iloc[:,idx].columns].fillna(train_num.iloc[:,idx].mean())
train_num.head(15)
sel_cat = (train.dtypes.values == 'O')

cat_idx = np.arange(0, len(train.columns))[sel_cat]

train_cat = train.iloc[:, cat_idx]



print('Number of Categorical Columns:  ', np.sum(sel_cat), '\n')

print('Indices for Categorical Columns:', cat_idx, '\n')

print('Names of Categorical Columns:\n', train_cat.columns.values)
train_cat.head(15)
train_cat = train_cat.fillna(train_cat.mode().iloc[0])
train_cat.nunique()
plt.figure(figsize=[12,4])



plt.subplot(1,2,1)

plt.hist(np.exp(y_train), bins=20, color='green', edgecolor='k')



plt.subplot(1,2,2)

plt.hist(y_train, bins=20, color='green', edgecolor='k')



plt.show()
test = test.drop(['PoolQC','MiscFeature','Alley','Fence'],axis = 1)
X_train=train

X_test = test.drop(['Id'], axis=1)



print('X_train shape:', X_train.shape)

print('y_train shape:', y_train.shape)

print('X_test shape: ', X_test.shape)
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

#print(train_proc[1,:])
encoded_names = preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names(train_cat.columns.values)

print(encoded_names[:20])



feature_names = np.concatenate([train_num.columns.values, encoded_names])

print(len(feature_names))
lr_pipe = Pipeline(

    steps = [

        ('preprocessor', preprocessor),

        ('regressor', LinearRegression())

    ]

)



lr_pipe.fit(X_train, y_train)

lr_pipe.score(X_train, y_train)
cv_results = cross_val_score(lr_pipe, X_train, y_train, cv=10, scoring='r2')



print('Results by fold:\n', cv_results, '\n')

print('Mean CV Score:', np.mean(cv_results))
cv_results = cross_val_score(lr_pipe, X_train, y_train, cv=10, scoring='neg_root_mean_squared_error')



print('Results by fold:\n', cv_results, '\n')

print('Mean CV Score:', np.mean(cv_results))
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