import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



import xgboost as xgb

from sklearn.pipeline import make_pipeline

from sklearn.feature_selection import SelectPercentile, chi2

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler





print(os.listdir("../input"))
data = pd.concat([

       pd.read_csv("../input/pageviews/pageviews.csv", parse_dates=["FEC_EVENT"]),

       pd.read_csv("../input/pageviews_complemento/pageviews_complemento.csv", parse_dates=["FEC_EVENT"])

])

#data
X_test = []

for c in data.drop(["USER_ID", "FEC_EVENT"], axis=1).columns:

    print("haciendo", c)

    temp = pd.crosstab(data.USER_ID, data[c])

    temp.columns = [c + "_" + str(v) for v in temp.columns]

    X_test.append(temp.apply(lambda x: x / x.sum(), axis=1))

X_test = pd.concat(X_test, axis=1)
data = data[data.FEC_EVENT.dt.month < 10]

X_train = []

for c in data.drop(["USER_ID", "FEC_EVENT"], axis=1).columns:

    print("haciendo", c)

    temp = pd.crosstab(data.USER_ID, data[c])

    temp.columns = [c + "_" + str(v) for v in temp.columns]

    X_train.append(temp.apply(lambda x: x / x.sum(), axis=1))

X_train = pd.concat(X_train, axis=1)
features = list(set(X_train.columns).intersection(set(X_test.columns)))

X_train = X_train[features]

X_test = X_test[features]
y_prev = pd.read_csv("../input/conversiones/conversiones.csv")

y_train = pd.Series(0, index=X_train.index)

idx = set(y_prev[y_prev.mes >= 10].USER_ID.unique()).intersection(

        set(X_train.index))

y_train.loc[list(idx)] = 1
params = {

    'selectpercentile__percentile': [15], 

    'logisticregression__C': np.logspace(-5, 5, 20)

}



pipe_ridge = make_pipeline(SelectPercentile(chi2), 

                           StandardScaler(), 

                           LogisticRegression(solver='sag', n_jobs=-1))

grid_ridge = GridSearchCV(pipe_ridge, params, scoring='roc_auc', cv=5, n_jobs=-1)

grid_ridge.fit(X_train.values, y_train)

grid_ridge.best_params_, grid_ridge.best_score_
results_ridge = pd.DataFrame(grid_ridge.cv_results_)

results_ridge
ridge_probs = grid_ridge.predict_proba(X_test)[:,0]
xgb_params = {'eta': 0.05,

              'gamma': 1.5,

              'subsample': 0.6,

              'colsample_bytree': 0.5,

              'max_depth': 4}



dtrain = xgb.DMatrix(X_train, label=y_train)

dtest = xgb.DMatrix(X_test)



cvres = xgb.cv(xgb_params, dtrain, num_boost_round=200, nfold=5, metrics=['auc'], stratified=True)

mdl = xgb.train(xgb_params, dtrain, num_boost_round=200)
gain = mdl.get_score(fmap='', importance_type='gain')

imp = pd.DataFrame(list(gain.items()), columns=['var', 'imp'])

imp.sort_values("imp", ascending=False)
xgb_probs = mdl.predict(dtest)
sm = pd.read_csv("../input/samplesubmission/sampleSubmission.csv")

sm['SCORE'] = ridge_probs * 0.5 + xgb_probs * 0.5

sm.to_csv("submission.csv", index=False)