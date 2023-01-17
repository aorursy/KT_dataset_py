!pip install cortado

!pip install xgboost
import pandas as pd



csvpath = "../input/airlinetrain1m/airlinetrain1m.csv"

df_pd = pd.read_csv(csvpath)

df_pd.head()
df_pd.info()
from scipy.sparse import coo_matrix

import numpy as np



covariates_xg = ["DepTime", "Distance"]

factors_xg = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]



sparse_covariates = list(map(lambda col: df_pd[col].astype(pd.SparseDtype("float32", 0.0)), covariates_xg))

sparse_factors = list(map(lambda col: pd.get_dummies(df_pd[col], prefix=col, sparse=True, dtype=np.float32), factors_xg))



data = pd.concat(sparse_factors + sparse_covariates, axis=1)

sparse_data = coo_matrix(data.sparse.to_coo()).tocsr()

label_xg = df_pd["dep_delayed_15min"].map({"N": 0, "Y": 1})
eta = 0.1

nrounds = 100

max_depth = 6
import xgboost as xgb

from datetime import datetime



start = datetime.now()

model = xgb.XGBClassifier(max_depth=max_depth, nthread=1, learning_rate=eta, tree_method="exact", n_estimators=nrounds)

model.fit(sparse_data, label_xg)

pred_xg = model.predict_proba(sparse_data)

end = datetime.now()

print("xgboost elapsed: {e}".format(e=(end - start)))

import cortado as cr



df_cr = cr.DataFrame.from_pandas(df_pd)
df_cr.covariates
df_cr.factors
deptime = cr.Factor.from_covariate(df_cr["DepTime"])

distance = cr.Factor.from_covariate(df_cr["Distance"])
deptime.levels[:5]
deptime = deptime.cached()

distance = distance.cached()
dep_delayed_15min = df_cr["dep_delayed_15min"]

label = cr.Covariate.from_factor(dep_delayed_15min, lambda level: level == "Y")

print(label)
factors = df_cr.factors + [deptime, distance]

factors.remove(dep_delayed_15min)
deptime.isordinal
df_cr["Month"].isordinal
start = datetime.now()

trees, pred_cr = cr.xgblogit(label, factors,  eta = eta, lambda_ = 1.0, gamma = 0.0, minh = 1.0, nrounds = nrounds, maxdepth = max_depth, slicelen=1000000)

end = datetime.now()

print("cortado elapsed: {e}".format(e=(end - start)))
from sklearn.metrics import roc_auc_score

y = label.to_array() # convert to numpy array

auc_cr = roc_auc_score(y, pred_cr) # cortado auc

auc_xg = roc_auc_score(y, pred_xg[:, 1]) # xgboost auc

print("cortado auc: {auc_cr}".format(auc_cr=auc_cr))

print("xgboost auc: {auc_xg}".format(auc_xg=auc_xg))

diff = np.max(np.abs(pred_xg[:, 1] - pred_cr))

print("max pred diff: {diff}".format(diff=diff))