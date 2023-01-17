import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px
import plotly.graph_objects as go
data = pd.read_excel("/kaggle/input/covid19/dataset.xlsx")
data
data.describe(include="all")
# Columns with no missing values: the first 6
data.columns[(data.isna().any() == False).values].values
(data.iloc[:, 6:].isna().sum() / len(data)).sort_values()[:20]
# No positive case when Influenza A is detected
data.groupby(["Influenza A", "SARS-Cov-2 exam result"])["Patient ID"].count()
# Records with NaN in all feature columns (i.e. no lab exams)
data.iloc[:, 6:].isna().all(axis=1).value_counts()
# Number of positive to negative cases among those with lab exams
data[data.iloc[:, 6:].isna().all(axis=1)]["SARS-Cov-2 exam result"].value_counts()
df = (data[data.iloc[:, 6:].isna().all(axis=1) == False] # Only patients that have at least one lab exam
                  .groupby(["Patient age quantile", "SARS-Cov-2 exam result"])["Patient ID"]
                  .count()
                  .reset_index()
                  .rename(columns={"Patient ID":"count"})
                  .sort_values("Patient age quantile")
     )

fig = go.Figure(data=[
    go.Bar(name='Positive', x=np.array(range(20)), y=df[df["SARS-Cov-2 exam result"] == "positive"]["count"]),
    go.Bar(name='Negative', x=np.array(range(20)), y=df[df["SARS-Cov-2 exam result"] == "negative"]["count"])
])
# Change the bar mode
fig.update_layout(barmode='group', 
                  xaxis_title="Age percentile", 
                  yaxis_title="Count",
                  title="Histogram: age distribution of patients with at least one lab exam, with and without COVID-19")
df = (data[data.iloc[:, 6:].isna().all(axis=1) == True] # Only patients that had no lab exams
                  .groupby(["Patient age quantile", "SARS-Cov-2 exam result"])["Patient ID"]
                  .count()
                  .reset_index()
                  .rename(columns={"Patient ID":"count"})
                  .sort_values("Patient age quantile")
     )

fig = go.Figure(data=[
    go.Bar(name='Positive', x=np.array(range(20)), y=df[df["SARS-Cov-2 exam result"] == "positive"]["count"]),
    go.Bar(name='Negative', x=np.array(range(20)), y=df[df["SARS-Cov-2 exam result"] == "negative"]["count"])
])
# Change the bar mode
fig.update_layout(barmode='group', 
                  xaxis_title="Age percentile", 
                  yaxis_title="Count",
                  title="Histogram: age distribution of patients with at least one lab exam, with and without COVID-19")
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import LabelEncoder
# Remove records that have null values for all exams 
data_w_exams = data[~data.iloc[:, 6:].isna().all(axis=1)]
y = data_w_exams["SARS-Cov-2 exam result"]
X = data_w_exams.iloc[:, 6:]
X["Patient age quantile"] = data_w_exams["Patient age quantile"]
X.loc[177, "Urine - pH"] = np.nan  # Substitute 'Não Realizado' for NaN
skip = {"Urine - pH":""} # No need to transform this feature into categorical

# Encode categorical features
les = []
for field in list(X.dtypes[X.dtypes == "object"].index.values):
#     print(field)
    if field in skip:
        X[field] = pd.to_numeric(X[field])
        continue
    le = LabelEncoder()
    idx = X[field].notnull()
    X.loc[idx, field] = pd.Series(le.fit_transform(X[idx][field]))
    X[field] = pd.to_numeric(X[field], errors="raise")
    les.append(le)
    
# Encode target field
ley = LabelEncoder()
y = ley.fit_transform(y)
dm_w_exams = xgb.DMatrix(data=X, label=y)
####################
##  Use cross-validation to evaluate different hyperparams of XGBoost models
####################

colsamples = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
depths = [3, 4, 5, 7, 9, 11]
alphas = [0, 0.1, 0.3, 1]
lrs = [0.1, 0.01]

best = None
for colsample in colsamples:
    for depth in depths:
        for alpha in alphas:
            for lr in lrs:
                params = {"objective":"binary:logistic",'colsample_bytree': colsample,'learning_rate': lr,
                                'max_depth': depth, 'alpha': alpha} # no improvement by changing scale_pos_weight here, needs more work

                cv_results = xgb.cv(dtrain=dm_w_exams, params=params, nfold=3,
                                    num_boost_round=500, early_stopping_rounds=10, 
                                    metrics="logloss", 
#                                     metrics="auc",
                                    as_pandas=True, seed=123)
                res = cv_results["test-logloss-mean"].iloc[-1]
#                 res = cv_results["test-auc-mean"].iloc[-1]
                if best is None:
                    best = (colsample, depth, alpha, lr, res)
                elif res < best[4]:
                    best = (colsample, depth, alpha, lr, res)
print("Best:\n", best)
params = {"objective":"binary:logistic",'colsample_bytree': best[0], 'learning_rate': best[3],
                                'max_depth': best[1], 'alpha': best[2]}

cv_results = xgb.cv(dtrain=dm_w_exams, params=params, nfold=3,
                    num_boost_round=700, early_stopping_rounds=10, 
                    metrics=["logloss"], as_pandas=True, seed=123)
num_rounds = len(cv_results)
cv_results
params = {"objective":"binary:logistic",'colsample_bytree': best[0], 'learning_rate': best[3],
                                'max_depth': best[1], 'alpha': best[2]}

model =  xgb.train(dtrain=dm_w_exams, params=params,
                    num_boost_round=num_rounds, 
                    )
# Top 20 most important features
res = list(model.get_score(importance_type="gain").items())
res.sort(key=lambda x: x[1])
res[::-1][:20]
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)

print(f"Number of samples in train set: {len(X_train)}\nNumber of samples in test set: {len(X_test)}")

params = {"objective":"binary:logistic",'colsample_bytree': best[0], 'learning_rate': best[3],
                                'max_depth': best[1], 'alpha': best[2]}

model =  xgb.train(dtrain=xgb.DMatrix(data=X_train, label=y_train), params=params,
                    num_boost_round=num_rounds)

preds = model.predict(xgb.DMatrix(data=X_test))

precision, recall, thresholds = precision_recall_curve(y_test, preds)
plt.figure(figsize=(10, 7))
plt.plot(precision, recall)
plt.ylabel("Precision")
plt.xlabel("Recall")
plt.title(f"Precision-Recall curve - PR-AUC: {auc(recall, precision):.3f}")

print("For comparison: AUC:", roc_auc_score(y_test, preds))
print("Accuracy: ", accuracy_score(y_test, np.around(preds)))
print("Precision: ", precision_score(y_test, np.around(preds)))
print("Recall: ", recall_score(y_test, np.around(preds)))
data_wo_exams = data[data.iloc[:, 6:].isna().all(axis=1)]
data_wo_exams
cols = ["Patient ID", "Patient age quantile", "SARS-Cov-2 exam result"]
data_wo_exams[cols].groupby(cols[1:]).count().groupby(level=0).apply(lambda x:
                                                 100 * x / float(x.sum()))