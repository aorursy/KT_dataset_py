import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import roc_auc_score
from IPython.display import display
%matplotlib inline 
data_path="../input"
data = pd.concat([pd.read_csv(os.path.join(data_path, "application_train.csv"), index_col="SK_ID_CURR"), pd.read_csv(os.path.join(data_path, "application_test.csv"), index_col="SK_ID_CURR")], axis=0)
data.head()
data.TARGET.fillna(-1).value_counts()
for c in data.select_dtypes(include="object"):
    data[c] = data[c].astype("category")
data.dtypes.value_counts()
bureau = pd.read_csv(os.path.join(data_path, "bureau.csv"))
bureau.head()
bureau_balance = pd.read_csv(os.path.join(data_path, "bureau_balance.csv")).join(bureau.set_index("SK_ID_BUREAU").SK_ID_CURR, on="SK_ID_BUREAU")
app = bureau[bureau.SK_ID_CURR == 100002]
display(app)
display(bureau_balance[bureau_balance.SK_ID_BUREAU.isin(app.SK_ID_BUREAU.unique())])
bureau_balance_dumies = pd.get_dummies(bureau_balance.STATUS)
bureau_balance_dumies["SK_ID_CURR"] = bureau_balance.SK_ID_CURR
bureau_balance_dumies = bureau_balance_dumies.groupby("SK_ID_CURR").sum()
irregular = ["bureau_balance_status_" + c for c in bureau_balance_dumies if c not in ["0", "C", "X"]]
very_irregular = ["bureau_balance_status_" + c for c in bureau_balance_dumies if c not in ["0", "1", "2", "C", "X"]]
bureau_balance_dumies.columns = ["bureau_balance_status_" + c for c in bureau_balance_dumies]
bureau_balance_dumies["total_balances"] = bureau_balance_dumies.sum(axis=1)
bureau_balance_dumies["irregular_bureau_balance"] = bureau_balance_dumies[irregular].sum(axis=1)
bureau_balance_dumies["irregular_bureau_balance_ratio"] = bureau_balance_dumies["irregular_bureau_balance"] / bureau_balance_dumies["total_balances"]
bureau_balance_dumies["very_irregular_bureau_balance"] = bureau_balance_dumies[very_irregular].sum(axis=1)
bureau_balance_dumies["very_irregular_bureau_balance_ratio"] = bureau_balance_dumies["very_irregular_bureau_balance"] / bureau_balance_dumies["total_balances"]
bureau_balance_dumies["super_irregular_bureau_balance_ratio"] = bureau_balance_dumies["bureau_balance_status_5"] / bureau_balance_dumies["total_balances"]
bureau_balance_dumies.head(100)
extended_data = [bureau_balance_dumies]
del bureau_balance_dumies, bureau_balance
for c in bureau.select_dtypes(["object"]):
    display(bureau[c].value_counts(dropna=False).to_frame())
bureau.groupby("SK_ID_CURR").CREDIT_CURRENCY.unique().apply(len).value_counts()
credits_status = pd.get_dummies(bureau.CREDIT_ACTIVE)
credits_status["SK_ID_CURR"] = bureau.SK_ID_CURR
credits_status = credits_status.groupby("SK_ID_CURR").sum(axis=0)

credits_status.columns = ["credits_status_" + "_".join(c.lower().split()) for c in credits_status]
total_credits = credits_status.sum(axis=1)

credits_status["sold_or_bad_credits"] = credits_status[["credits_status_bad_debt", "credits_status_sold"]].sum(axis=1)

ratios = pd.concat(((credits_status[c] / total_credits).rename(c + "_ratio") for c in credits_status), axis=1)

credits_status["total_credits"] = total_credits
credits_status = credits_status.join(ratios)
credits_status
extended_data.append(credits_status)
del credits_status
credits_types = pd.get_dummies(bureau.CREDIT_TYPE)
credits_types["SK_ID_CURR"] = bureau.SK_ID_CURR
credits_types = credits_types.groupby("SK_ID_CURR").sum(axis=0)
credits_types.columns = ["credits_types_" + "_".join(c.lower().split()) for c in credits_types]
credits_types
extended_data.append(credits_types)
del credits_types
bureau.DAYS_ENDDATE_FACT.fillna(bureau.DAYS_CREDIT_ENDDATE, inplace=True)
bureau.DAYS_CREDIT_ENDDATE.fillna(bureau.DAYS_ENDDATE_FACT, inplace=True)
bureau.DAYS_ENDDATE_FACT.fillna(0, inplace=True)
bureau.DAYS_CREDIT_ENDDATE.fillna(0, inplace=True)
bureau["days_end_credit_advancement"] = bureau.DAYS_CREDIT_ENDDATE - bureau.DAYS_ENDDATE_FACT
max_cols = ["CREDIT_DAY_OVERDUE", "AMT_CREDIT_MAX_OVERDUE", "CNT_CREDIT_PROLONG", "days_end_credit_advancement"]
bureau_agg_data = bureau[max_cols + ["SK_ID_CURR"]].fillna(0).groupby("SK_ID_CURR").max()
bureau_agg_data.columns = ["max_" + "_".join(c.lower().split()) for c in bureau_agg_data]
display(bureau_agg_data)

extended_data.append(bureau_agg_data)
del bureau_agg_data
sum_cols = ["CREDIT_DAY_OVERDUE", "AMT_CREDIT_MAX_OVERDUE", "CNT_CREDIT_PROLONG", "days_end_credit_advancement", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_LIMIT", "AMT_CREDIT_SUM_OVERDUE"]
bureau_agg_data = bureau[sum_cols + ["SK_ID_CURR"]].fillna(0).groupby("SK_ID_CURR").sum(axis=0)
bureau_agg_data["total_outstanding_debt_ratio"] = bureau_agg_data.AMT_CREDIT_SUM_DEBT / bureau_agg_data.AMT_CREDIT_SUM
bureau_agg_data.columns = ["sum_" + "_".join(c.lower().split()) for c in bureau_agg_data]
display(bureau_agg_data)

extended_data.append(bureau_agg_data)
del bureau_agg_data
sum_cols = ["AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT"]
active_bureau =  bureau[bureau.CREDIT_ACTIVE == "Active"][sum_cols + ["SK_ID_CURR"]].groupby("SK_ID_CURR").sum(axis=0)
active_bureau.columns = ["total_active_debts", "outstanding_active_debts"]
display(active_bureau)
extended_data.append(active_bureau)
del active_bureau
data = data.join(pd.concat(extended_data, axis=1))
display(data)
del extended_data
data["credit_goodsprice_ratio"] = data.AMT_CREDIT / data.AMT_GOODS_PRICE
data["annuity_income_ratio"] = data.AMT_ANNUITY / data.AMT_INCOME_TOTAL
data["annuity_credit_ratio"] = data.AMT_ANNUITY / data.AMT_CREDIT
data["total_new_credit_increase"] = data.AMT_CREDIT + data.outstanding_active_debts
data["credit_increase"] = data.AMT_CREDIT / data.total_new_credit_increase
data["total_new_credit_income_ratio"] = data.total_new_credit_increase / data.AMT_INCOME_TOTAL
data.head(100)
from lightgbm import LGBMClassifier
def train_model(train, target, nl, X_test, folds=5):
    test_probs = []
    for i in range(folds):
        X_valid = train.sample(frac=1/folds)
        y_valid = X_valid[target]
        X_valid = X_valid.drop(target, axis=1)
        
        X_train = train.drop(X_valid.index)
        y_train = X_train[target]
        X_train = X_train.drop(target, axis=1)

        learner = LGBMClassifier(n_estimators=10000, num_leaves=nl)
        learner.fit(X_train, y_train,  early_stopping_rounds=10, eval_metric="auc", verbose=50,
                    eval_set=[(X_train, y_train),
                              (X_valid, y_valid)])
        probs = pd.Series(learner.predict_proba(X_test)[:, -1], index=X_test.index, name="fold_" + str(i))
        test_probs.append(probs)
    return pd.concat(test_probs, axis=1).mean(axis=1)
test = data[data.TARGET.notnull()].sample(frac=0.1)
train = data[data.TARGET.notnull()].drop(test.index)
X_test = test.drop("TARGET", axis=1)
y_test = test.TARGET

nls = [2 ** i for i in [4, 5, 6]]
res = pd.Series([np.nan] * len(nls), index=nls, name="ROC_AUC")
for nl in nls:
    print("*"*10, nl, "*"*10)
    probs = train_model(train, "TARGET", nl, X_test, 10)
    res.loc[nl] = roc_auc_score(y_test, probs)
    print("ROC_AUC para {nl} hojas: {res:.4f}".format(nl=nl, res=res.loc[nl]))
res.to_frame()
train = data[data.TARGET.notnull()]
X_test = data.drop(train.index).drop("TARGET", axis=1)
nl = res.idxmax()
train_model(train, "TARGET", nl, X_test).rename("TARGET").to_csv("submission_{nl}.csv".format(nl=nl), header=True)