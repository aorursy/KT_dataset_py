!pip install cortado
import pandas as pd



csvpath = "../input/airlinetrain1m.csv"

df_pd = pd.read_csv(csvpath)

df_pd.head()
df_pd.info()
import cortado as cr



df_cr = cr.DataFrame.from_pandas(df_pd)
df_cr.covariates
df_cr.factors
deptime = cr.Factor.from_covariate(df_cr["DepTime"])

distance = cr.Factor.from_covariate(df_cr["Distance"])
print(deptime)
deptime.levels[:5]
deptime = deptime.cached()

distance = distance.cached()
dep_delayed_15min = df_cr["dep_delayed_15min"]

print(dep_delayed_15min)
label = cr.Covariate.from_factor(dep_delayed_15min, lambda level: level == "Y")

print(label)


label = cr.Covariate.from_factor(dep_delayed_15min, {"Y" : 1, "N": 0})

print(label)
factors = df_cr.factors + [deptime, distance]

factors.remove(dep_delayed_15min)

deptime.isordinal
df_cr["Month"].isordinal
trees, pred = cr.xgblogit(label, factors,  eta = 0.1, lambda_ = 1.0, gamma = 0.0, minh = 1.0, nrounds = 10, maxdepth = 4, slicelen=1000000)
from sklearn.metrics import roc_auc_score

y = label.to_array() # convert to numpy array

auc = roc_auc_score(y, pred)

print(auc)

print(pred[:5])