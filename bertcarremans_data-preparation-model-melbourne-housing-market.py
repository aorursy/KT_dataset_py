import pandas as pd

import numpy as np

import os

import warnings; warnings.simplefilter('ignore')

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import FunctionTransformer

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from scipy import stats

from sklearn.externals import joblib
RAW_DATA_FILE = 'melb_data.csv'

# check if the notebook is running on Kaggle or locally

if os.getcwd() == '/kaggle/working':

    RAW_DATA_PATH = '../input'

else:

    RAW_DATA_PATH = os.path.join('../data','raw')
def load_data(raw_data_path, raw_data_file):

    cols_to_use = ['Rooms','Price','Method','Date','Distance','Propertycount','Bedroom2','Bathroom','Car','Landsize','BuildingArea','YearBuilt','Lattitude','Longtitude']

    df = pd.read_csv(os.path.join(raw_data_path, raw_data_file), usecols =[i for i in cols_to_use])

    df['Date'] = pd.to_datetime(df['Date'])

    return df



df = load_data(RAW_DATA_PATH, RAW_DATA_FILE)
train_set, test_set = train_test_split(df, test_size=0.2, random_state=38, shuffle=True)

print(train_set.shape)

print(test_set.shape)
def separate_target_input(df):

    housing = train_set.drop('Price', axis=1)

    housing_labels = train_set['Price'].copy()

    return housing, housing_labels



X_train, y_train = separate_target_input(train_set)

X_test, y_test = separate_target_input(test_set)
def feat_extract(X):

    X['other_rooms'] = X.Rooms - X.Bathroom

    X['surface_per_room'] = X.BuildingArea/X.Rooms

    X['perc_built'] = X.BuildingArea/X.Landsize

    X['house_age'] = X.Date.dt.year - X.YearBuilt

    return X
num_pipeline = Pipeline(steps=[

    ('imputer', Imputer(strategy='median')),

    ('scaler', StandardScaler())

])
cat_pipeline = Pipeline(steps=[

    ('ohe', OneHotEncoder(handle_unknown='ignore', categories='auto', sparse=False))

])
cat_vars = X_train.select_dtypes(include=[object]).columns.tolist()

num_vars = X_train.select_dtypes(include=[np.number]).columns.tolist()



full_pipeline = Pipeline(steps=[

    ('feat_extract', FunctionTransformer(feat_extract, validate=False)),

    ('union', ColumnTransformer(transformers=[

        ('num_pipeline', num_pipeline, num_vars),

        ('cat_pipeline', cat_pipeline, cat_vars)],

        remainder='drop')

    )

])

X_train_prep = full_pipeline.fit_transform(X_train)
def display_scores(scores):

    print('Scores:', scores)

    print('Mean:', np.mean(scores))

    print('Standard Deviation:', np.std(scores))
linreg = LinearRegression()

scores = cross_val_score(linreg, X_train_prep, y_train, cv=10, scoring='neg_mean_squared_error')

linreg_scores = np.sqrt(-scores)

display_scores(linreg_scores)
dectree = DecisionTreeRegressor()

scores = cross_val_score(dectree, X_train_prep, y_train, cv=10, scoring='neg_mean_squared_error')

dectree_scores = np.sqrt(-scores)

display_scores(dectree_scores)
forest = RandomForestRegressor(n_estimators=30)

scores = cross_val_score(forest, X_train_prep, y_train, cv=10, scoring='neg_mean_squared_error')

forest_scores = np.sqrt(-scores)

display_scores(forest_scores)
svr_rbf = SVR(kernel='rbf', gamma='auto')

scores = cross_val_score(svr_rbf, X_train_prep, y_train, cv=10, scoring='neg_mean_squared_error')

svr_rbf_scores = np.sqrt(-scores)

display_scores(svr_rbf_scores)
ada = AdaBoostRegressor()

scores = cross_val_score(ada, X_train_prep, y_train, cv=10, scoring='neg_mean_squared_error')

ada_scores = np.sqrt(-scores)

display_scores(ada_scores)
param_grid = {

    'n_estimators': [200],

    'max_features': [0.5,1.0],

    'bootstrap': [True, False]

}



rf = RandomForestRegressor()

gs = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')



gs.fit(X_train_prep, y_train)
gs.best_params_
np.sqrt(-gs.best_score_)
for score, params in zip(gs.cv_results_['mean_test_score'],gs.cv_results_['params']):

    print(np.sqrt(-score), params)
extra_vars = ['other_rooms', 'surface_per_room', 'perc_built', 'house_age']

cat_ohe_vars = list(OneHotEncoder().fit(X_train[cat_vars]).categories_[0])

prep_vars = extra_vars + num_vars + cat_ohe_vars

feature_importances = gs.best_estimator_.feature_importances_

sorted(zip(feature_importances, prep_vars), reverse=True)
param_dist = {

    'n_estimators': stats.randint(low=10, high=200),

    'max_features': stats.randint(low=2, high=X_train_prep.shape[1]),

    'bootstrap': [True, False]

}



rs = RandomizedSearchCV(estimator=RandomForestRegressor(), param_distributions=param_dist, cv=5, n_jobs=-1, scoring='neg_mean_squared_error', n_iter=50)

rs.fit(X_train_prep, y_train)
rs.best_params_
np.sqrt(-rs.best_score_)
for score, params in zip(rs.cv_results_['mean_test_score'],rs.cv_results_['params']):

    print(np.sqrt(-score), params)
full_pipeline_with_model = Pipeline([

    ('prep', full_pipeline),

    ('model', RandomForestRegressor(**rs.best_params_))

])



prep_param_grid = {

    'prep__union__num_pipeline__imputer__strategy': ['most_frequent', 'median', 'mean']

}



gs2 = GridSearchCV(full_pipeline_with_model, param_grid=prep_param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')

gs2.fit(X_train, y_train)
gs2.best_params_
preds_test = gs2.best_estimator_.predict(X_test)

mse_test = mean_squared_error(y_test, preds_test)

print('RMSE on test set', np.sqrt(mse_test))
squared_errors = (y_test - preds_test)**2

m = len(squared_errors)

confidence = 0.95

ci = np.sqrt(stats.t.interval(confidence, m - 1, loc=np.mean(squared_errors), scale=stats.sem(squared_errors)))

ci
final_pipeline = Pipeline([

    ('prep', full_pipeline),

    ('model', RandomForestRegressor(**rs.best_params_))

])



final_pipeline.fit(X_train, y_train)



joblib.dump(final_pipeline, 'final_pipeline.pkl')
final_pipeline_loaded = joblib.load('final_pipeline.pkl')

final_pipeline_loaded.predict(X_test)