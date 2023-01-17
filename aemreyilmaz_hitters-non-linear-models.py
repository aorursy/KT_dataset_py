import warnings

warnings.simplefilter(action='ignore')



import pandas as pd

import numpy as np

from sklearn.metrics import mean_squared_error

from sklearn.neighbors import LocalOutlierFactor

from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler, Normalizer

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor

!pip install xgboost

import xgboost

from xgboost import XGBRegressor

!pip install lightgbm

from lightgbm import LGBMRegressor

!pip install catboost

from catboost import CatBoostRegressor
hitters = pd.read_csv('../input/hitters-baseball-data/Hitters.csv')

df = hitters.copy()

df.head()
df['HitRatio'] = df['Hits'] / df['AtBat']

df['RunRatio'] = df['HmRun'] / df['Runs']

df['CHitRatio'] = df['CHits'] / df['CAtBat']

df['CRunRatio'] = df['CHmRun'] / df['CRuns']



df['Avg_AtBat'] = df['CAtBat'] / df['Years']

df['Avg_Hits'] = df['CHits'] / df['Years']

df['Avg_HmRun'] = df['CHmRun'] / df['Years']

df['Avg_Runs'] = df['CRuns'] / df['Years']

df['Avg_RBI'] = df['CRBI'] / df['Years']

df['Avg_Walks'] = df['CWalks'] / df['Years']

df['Avg_PutOuts'] = df['PutOuts'] / df['Years']

df['Avg_Assists'] = df['Assists'] / df['Years']

df['Avg_Errors'] = df['Errors'] / df['Years']
le = LabelEncoder()

df['League'] = le.fit_transform(df['League'])

df['Division'] = le.fit_transform(df['Division'])

df['NewLeague'] = le.fit_transform(df['NewLeague'])
df.dropna(inplace = True)
clf = LocalOutlierFactor(n_neighbors = 20, contamination = 0.1)

clf.fit_predict(df)

df_scores = clf.negative_outlier_factor_

np.sort(df_scores)[0:15]
thrs = np.sort(df_scores)[3]

thrs
df.drop(df[df_scores < thrs].index, inplace = True)
dfx = df.copy()
dfx = dfx.drop(['AtBat','Hits','HmRun','Runs','RBI','Salary','League','Division','NewLeague'], axis = 1)

# I dropped 'Salary' since it's dependent variable

# I dropped 'League', 'Division' and 'NewLeague' in order to perform a better scaling

# I dropped the others because some of the new assigned variables are better representatives
cols = dfx.columns

scaler = RobustScaler()

X = scaler.fit_transform(dfx)

X = pd.DataFrame(X, columns = cols)

y = df[['Salary']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state  = 46)
def model_func(alg):

    if alg == CatBoostRegressor:

        model = alg(verbose = False).fit(X_train, y_train)

    else:

        model = alg().fit(X_train, y_train)

    y_pred = model.predict(X_test)

    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))

    model_name = alg.__name__

    print(model_name, 'RMSE: ', RMSE)
def cv_func(alg, **param):

    

    model = alg().fit(X_train, y_train)

    params = {}

    for key, value in param.items():

        params[key] = value

    

    cv_model = GridSearchCV(model, params, cv = 10, verbose = 2, n_jobs = -1).fit(X_train, y_train)

    print(cv_model.best_params_)

    tuned_model = alg(**cv_model.best_params_).fit(X_train, y_train)

    y_pred = tuned_model.predict(X_test)

    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))

    model_name = alg.__name__

    print(model_name, 'RMSE: ', RMSE)
models = [KNeighborsRegressor, SVR, MLPRegressor, GradientBoostingRegressor, DecisionTreeRegressor, RandomForestRegressor, XGBRegressor, LGBMRegressor, CatBoostRegressor]

for model in models:

    model_func(model)
cv_func(alg = KNeighborsRegressor, n_neighbors = np.arange(2,30,1))
cv_func(alg = SVR, C = [0.01, 0.02, 0.2, 0.1, 0.5, 0.8, 1])
cv_func(alg = MLPRegressor, alpha = [0.1, 0.02, 0.01, 0.001, 0.0001], hidden_layer_sizes = [(10,20), (5,5), (100,100)])
cv_func(alg = GradientBoostingRegressor, max_depth = [3,5,8], learning_rate = [0.001,0.01,0.1], n_estimators = [100,200,500,1000], subsample = [0.3,0.5,0.8,1])
cv_func(alg = DecisionTreeRegressor, max_depth = [2,3,4,5,10,20], min_samples_split = [2,5,10,20,30,50])
cv_func(alg = RandomForestRegressor, max_depth = [5,10,None], max_features = [5,10,15,20], n_estimators = [500, 1000], min_samples_split = [2,5,20,30])
cv_func(alg = XGBRegressor, max_depth = [2,3,4,5,8], learning_rate = [0.1,0.5,0.01], n_estimators = [100,200,500,1000], colsample_bytree = [0.4,0.7,1])
cv_func(alg = LGBMRegressor, max_depth = [1,2,3,4,5,6,7,8,9,10], n_estimators = [20,40,100,200,500,1000], learning_rate = [0.1,0.01,0.5,1])
cv_func(alg = CatBoostRegressor, iterations = [200], learning_rate = [0.02, 0.03, 0.05], depth = [8, 10])
# final model can differ after each run. Gradient Boosting Regressor was the best when I ran the code, so I created final model manually using its stats.

final_model = GradientBoostingRegressor(learning_rate = 0.01, max_depth = 5, n_estimators = 500, subsample = 0.8)
final_model