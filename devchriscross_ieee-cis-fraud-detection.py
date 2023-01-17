import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


import os
import gc

import catboost

from catboost import CatBoostClassifier
X_transaction = pd.read_csv("../input/ieee-fraud-detection/train_transaction.csv")
X_identity = pd.read_csv("../input/ieee-fraud-detection/train_identity.csv", usecols=["TransactionID"])
X_transaction.sample(5)
id_with_identity = X_identity["TransactionID"].unique().tolist()
X_transaction["with_identity"] = X_transaction["TransactionID"].isin(id_with_identity)
X_transaction.sample(5)
X_transaction.with_identity.value_counts()
X_transaction.isFraud.value_counts()
X_transaction["isFraud"] = X_transaction["isFraud"].astype(bool) 
sns.countplot(x="with_identity", data=X_transaction[X_transaction.isFraud])
sns.countplot(x="with_identity", data=X_transaction[~X_transaction.isFraud])
class_ratio = [len(X_transaction[X_transaction.isFraud & X_transaction.with_identity]),
                len(X_transaction[X_transaction.isFraud & ~X_transaction.with_identity]),
                len(X_transaction[~X_transaction.isFraud & X_transaction.with_identity]),
                len(X_transaction[~X_transaction.isFraud & ~X_transaction.with_identity])]
class_ratio = class_ratio / np.min(class_ratio)
class_ratio
X_transaction["isFraud"] = X_transaction["isFraud"].astype(bool) 
X_transaction = X_transaction.sort_values(["TransactionDT"])

train_test_idx = int(len(X_transaction)*0.75)
X_train_ = X_transaction.loc[:train_test_idx,:]
X_test = X_transaction.loc[train_test_idx:,:]
print("Training set (with validation):", len(X_train_), "Test set:", len(X_test))

train_valid_idx = int(len(X_train_)*0.75)
X_train = X_train_.loc[:train_valid_idx,:]
X_valid = X_train_.loc[train_valid_idx:,:]
print("Training set:", len(X_train), "Validation set:", len(X_valid))

del X_transaction
gc.collect()
# seed = 1
# strat_group1 = X_transaction[X_transaction.isFraud & X_transaction.with_identity]
# strat_group2 = X_transaction[X_transaction.isFraud & ~X_transaction.with_identity]
# strat_group3 = X_transaction[~X_transaction.isFraud & X_transaction.with_identity]
# strat_group4 = X_transaction[~X_transaction.isFraud & ~X_transaction.with_identity]
# X_transaction["isFraud"] = X_transaction["isFraud"].astype(bool) 
# strat_groups = [X_transaction.isFraud, ~X_transaction.isFraud]

# X_train = pd.DataFrame()
# X_valid = pd.DataFrame()

# for group in strat_groups:
#     X_train_partial = X_transaction[group].sample(frac=0.75, random_state=seed)
#     X_valid_partial = X_transaction[group].sample(frac=0.25, random_state=seed)
    
#     X_train = pd.concat([X_train, X_train_partial])
#     X_valid = pd.concat([X_valid, X_valid_partial])

# del X_transaction, strat_groups
# , strat_group1, strat_group2, strat_group3, strat_group4
# gc.collect()
X_train = X_train.reset_index(drop=True)
X_train = X_train.sort_values(["TransactionDT"])

X_valid = X_valid.reset_index(drop=True)
X_valid = X_valid.sort_values(["TransactionDT"])

X_test = X_test.reset_index(drop=True)
X_test = X_test.sort_values(["TransactionDT"])
def model_baseline(X_train, X_valid):
    seed = 1
    cat_str_features = ["ProductCD", "P_emaildomain", "R_emaildomain", "card4", "card6"]
    cat_str_features.extend(["M"+str(i+1) for i in range(9)])

    cat_num_features = ["addr1", "addr2"]
    cat_num_features.extend(["card"+str(i+1) for i in range(6) if (i!=3 and i!=5)])

    cat_features = cat_str_features + cat_num_features
    
    X_train[cat_num_features] = X_train[cat_num_features].astype("category")
    X_valid[cat_num_features] = X_valid[cat_num_features].astype("category")

    X_train[cat_features] = X_train[cat_features].astype("str").fillna("nan").astype("category")
    X_valid[cat_features] = X_valid[cat_features].astype("str").fillna("nan").astype("category")

    X_train[cat_features].isnull().mean()
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_class_weight

    scaler = StandardScaler()
    num_features = X_train.select_dtypes("number").drop(columns=["TransactionID", "TransactionDT"]).columns
    print(num_features.tolist())

    scaled_features = pd.DataFrame(scaler.fit_transform(X_train[num_features]), columns=num_features)
    X_train = X_train.drop(columns=num_features)
    X_train = pd.concat([X_train, scaled_features], axis=1)

    scaled_features = pd.DataFrame(scaler.transform(X_valid[num_features]), columns=num_features)
    X_valid = X_valid.drop(columns=num_features)
    X_valid = pd.concat([X_valid, scaled_features], axis=1)

    y_train = X_train.isFraud.astype("uint8").to_numpy()
    X_train = X_train.drop(columns=["TransactionID", "TransactionDT", "isFraud"])

    y_valid = X_valid.isFraud.astype("uint8").to_numpy()
    X_valid = X_valid.drop(columns=["TransactionID", "TransactionDT", "isFraud"])

    c_weight = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)

    del scaled_features
    gc.collect()
    
    cboost = CatBoostClassifier(loss_function="Logloss", random_seed=seed, class_weights=c_weight, cat_features=cat_features, iterations=1000)
    cboost.fit(X_train, y_train, cat_features=cat_features, eval_set=catboost.Pool(X_valid, label=y_valid, cat_features=cat_features), plot=True, early_stopping_rounds=100)

    from sklearn.metrics import classification_report
    print(classification_report(y_valid, cboost.predict(X_valid).reshape(-1)))
    
    from sklearn.metrics import roc_auc_score
    print("ROC-AUC score:", roc_auc_score(y_valid, cboost.predict(X_valid).reshape(-1), average="weighted"))
model_baseline(X_train.copy(), X_valid.copy())
#     precision    recall  f1-score   support
# 0       0.99      0.93      0.96    106379
# 1       0.29      0.70      0.41      4348

#     accuracy                           0.92    110727
#    macro avg       0.64      0.81      0.69    110727
# weighted avg       0.96      0.92      0.94    110727

# ROC-AUC score: 0.8141551715515302
# cboost2.best_score_
# sorted(tuple(zip(X_train.columns, cboost2.feature_importances_)), key=lambda x: x[1], reverse=True)
X_train.sample(10)
def add_day_count(X_train, min_transactiondt, day_of_week=True):
    seconds_per_day = 60*60*24
    seconds_per_week = seconds_per_day*7
    seconds_per_semimonthly = seconds_per_day*15
    seconds_per_month = seconds_per_day*30

    datetime_bin_day = np.arange(min_transactiondt, np.max(X_train["TransactionDT"])+seconds_per_day, seconds_per_day)
    datetime_bin_week = np.arange(min_transactiondt, np.max(X_train["TransactionDT"])+seconds_per_week, seconds_per_week)
    datetime_bin_semimonthly = np.arange(min_transactiondt, np.max(X_train["TransactionDT"])+seconds_per_semimonthly, seconds_per_semimonthly)
    datetime_bin_month = np.arange(min_transactiondt, np.max(X_train["TransactionDT"])+seconds_per_month, seconds_per_month)

    X_train["Day"] = pd.cut(X_train["TransactionDT"], bins=datetime_bin_day, labels=range(len(datetime_bin_day)-1)).astype(np.uint8).add(1)
    X_train["Week"] = pd.cut(X_train["TransactionDT"],  bins=datetime_bin_week, labels=range(len(datetime_bin_week)-1)).astype(np.uint8).add(1)
    X_train["SemiMonthly"] = pd.cut(X_train["TransactionDT"],  bins=datetime_bin_semimonthly, labels=range(len(datetime_bin_semimonthly)-1)).astype(np.uint8).add(1)
    X_train["Month"] = pd.cut(X_train["TransactionDT"],  bins=datetime_bin_month, labels=range(len(datetime_bin_month)-1)).astype(np.uint8).add(1)
    
    if day_of_week:
        DAYS_IN_WEEK = 7 
        X_train["DayOfWeek"] = DAYS_IN_WEEK
        for i in range(1, DAYS_IN_WEEK):
            X_train.loc[(X_train.Day % DAYS_IN_WEEK) == i, "DayOfWeek"] = i

        WEEKS_IN_MONTH = 4 
        X_train["WeekOfMonth4"] = WEEKS_IN_MONTH
        for i in range(1, WEEKS_IN_MONTH):
            X_train.loc[(X_train.Week % WEEKS_IN_MONTH) == i, "WeekOfMonth4"] = i

        WEEKS_IN_MONTH = WEEKS_IN_MONTH + 1
        X_train["WeekOfMonth5"] = WEEKS_IN_MONTH
        for i in range(1, WEEKS_IN_MONTH):
            X_train.loc[(X_train.Week % WEEKS_IN_MONTH) == i, "WeekOfMonth5"] = i
    
    return X_train
MIN_TRANSACTION_DT = np.min(X_train["TransactionDT"])-1
X_train = add_day_count(X_train, MIN_TRANSACTION_DT)
X_train.sample(5)
def fraud_occurrences(transactionInterval): 
    fraud_count_week = X_train.groupby([transactionInterval, "isFraud"])["isFraud"].count().to_frame()\
            .rename(columns={"isFraud": "count"}).reset_index()

    total_count = fraud_count_week.groupby([transactionInterval])["count"].sum().to_frame()
    fraud_count_week = fraud_count_week[fraud_count_week.isFraud].reset_index(drop=True)
    
    total_num_fraud = np.sum(fraud_count_week["count"])
    fraud_count_week["cum_count"] = fraud_count_week["count"].expanding().sum()
    fraud_count_week["cum_count"] = fraud_count_week["cum_count"].divide(total_num_fraud)
    fraud_count_week["count"] = fraud_count_week["count"].divide(total_count["count"])
    
    sns.set_style("darkgrid")
    plt.figure(figsize=(15,5))
    ax = sns.lineplot(y="count", x=transactionInterval, data=fraud_count_week)
    sns.scatterplot(y="count", x=transactionInterval, data=fraud_count_week)
    ax.set(xlabel=transactionInterval, ylabel="Percentage of Fraud", title="Percentage of Fraud per " + transactionInterval)
    # percentage of reported fraud overall transactions made
    
    plt.figure(figsize=(15,5))
    ax = sns.lineplot(y="cum_count", x=transactionInterval, data=fraud_count_week)
    sns.scatterplot(y="cum_count", x=transactionInterval, data=fraud_count_week)
    ax.set(xlabel=transactionInterval, ylabel="Cumulative Percentage of Fraud", title="Cumulative Percentage of total Frauds per " + transactionInterval)
fraud_occurrences("Day")
fraud_occurrences("Week")
fraud_occurrences("SemiMonthly")
fraud_occurrences("Month")
def plot_category_trends(X_train, col):
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(1, 3, figsize=(25,5))
    
    order = X_train[col].unique().tolist()
    
    sns.countplot(x=col, data=X_train, ax=ax[0], order=order)
    ax[0].set(xlabel=col, ylabel="Transaction Count", title="Total Transactions per " + col)
    
    count_per_category = X_train[X_train.isFraud].groupby([col]).TransactionID.count()
    fraud_per_category = count_per_category / X_train.groupby([col]).TransactionID.count()
    fraud_per_total = count_per_category / X_train[X_train.isFraud]["isFraud"].astype(bool).sum()
    
#     print(count_per_category)
#     print(X_train.groupby([col]).TransactionID.count())
    
    sns.barplot(y=fraud_per_category, x=fraud_per_category.index, ax=ax[1], order=order)
    ax[1].set(xlabel=col, ylabel="Percentage of Fraud", title="Percentage of Fraud per " + col)
    
    sns.barplot(y=fraud_per_total, x=fraud_per_total.index, ax=ax[2], order=order)
    ax[2].set(xlabel=col, ylabel="Percentage of Fraud", title="Distribution of Fraud Occurences per " + col)
plot_category_trends(X_train, "DayOfWeek")
plot_category_trends(X_train, "WeekOfMonth4")
plot_category_trends(X_train, "WeekOfMonth5")
X_train.sample(5)
X_train["TransactionAmt"].describe()
X_train.sort_values(["TransactionAmt"]).tail(10)
X_train["isFraud"] = X_train["isFraud"].astype("category")
plt.figure(figsize=(15,5))
sns.boxenplot(x="TransactionAmt", y="isFraud", data=X_train[X_train.TransactionAmt < X_train.TransactionAmt.max()])
plt.figure(figsize=(15,5))
amount_quantile_99 = X_train["TransactionAmt"].quantile(0.99)
amount_with_less_99q = X_train[X_train.TransactionAmt <= amount_quantile_99]
amount_with_more_99q = X_train[X_train.TransactionAmt > amount_quantile_99]

print("Number of outliers:", len(amount_with_more_99q))

X_train["isFraud"] = X_train["isFraud"].astype("category")
sns.boxenplot(x="TransactionAmt", y="isFraud", data=amount_with_less_99q)
plt.figure(figsize=(15,5))
sns.pointplot(x="TransactionAmt", y="isFraud", data=X_train[X_train.TransactionAmt < X_train.TransactionAmt.max()], color="b")
sns.pointplot(x="TransactionAmt", y="isFraud", data=amount_with_less_99q, color="r")
amount_with_more_99q.isFraud.value_counts()
# plotting outliers
plt.figure(figsize=(15,5))
sns.boxenplot(x="TransactionAmt", y="isFraud", 
              data=amount_with_more_99q[amount_with_more_99q.TransactionAmt < np.max(amount_with_more_99q["TransactionAmt"])])
amount_with_less_99q["isFraud"] = amount_with_less_99q["isFraud"].astype(bool)
plt.figure(figsize=(15,5))
sns.lineplot(y="TransactionAmt", x="Week", data=amount_with_less_99q, color="b", ci=None)
sns.lineplot(y="TransactionAmt", x="Week", data=amount_with_less_99q[amount_with_less_99q.isFraud], color="r")
sns.lineplot(y="TransactionAmt", x="Week", data=amount_with_less_99q[~amount_with_less_99q.isFraud], color="g")
plt.figure(figsize=(15,5))
sns.lineplot(y="TransactionAmt", x="Week", data=amount_with_less_99q[amount_with_less_99q.isFraud], color="r", ci="sd")
sns.lineplot(y="TransactionAmt", x="Week", data=amount_with_less_99q[~amount_with_less_99q.isFraud], color="g", ci="sd")
plt.figure(figsize=(20,8))
sns.boxenplot(y="TransactionAmt", x="Week", data=amount_with_less_99q, hue="isFraud")
def add_trx_features(X_train):
    avg_trx_week = X_train.groupby(["Week"]).TransactionAmt.mean().to_dict()
    X_train["avg_trx_week"] = X_train.Week.map(avg_trx_week).astype(np.float64)
    X_train["offset_from_mean_week"] = X_train.TransactionAmt.subtract(X_train.avg_trx_week).abs()
    X_train["offset_from_mean_week_2"] = X_train["offset_from_mean_week"].pow(2)

    std_trx_week = X_train.groupby(["Week"]).TransactionAmt.std().to_dict()
    X_train["std_trx_week"] = X_train.Week.map(std_trx_week).astype(np.float64)
    X_train["std_from_mean_week"] = X_train.TransactionAmt.divide(X_train.std_trx_week)

    return X_train
amount_with_less_99q = add_trx_features(amount_with_less_99q)
amount_with_less_99q["isFraud"] = amount_with_less_99q["isFraud"].astype("category")
plt.figure(figsize=(15,5))
sns.boxenplot(x="std_from_mean_week", y="isFraud", data=amount_with_less_99q)
plt.figure(figsize=(15,5))
sns.boxenplot(x="offset_from_mean_week", y="isFraud", data=amount_with_less_99q)
def add_trx_rolling_features(X_train):
    avg_num_trx_day = int(X_train.groupby(["Day"])["isFraud"].count().mean())
    avg_num_trx_week = int(X_train.groupby(["Week"])["isFraud"].count().mean())
    X_train["rolling_mean_" + str(avg_num_trx_week)] = X_train.TransactionAmt.rolling(avg_num_trx_week, min_periods=avg_num_trx_day).mean()
    X_train["rolling_std_" + str(avg_num_trx_week)] = X_train.TransactionAmt.rolling(avg_num_trx_week, min_periods=avg_num_trx_day).std()
    X_train["rolling_min_" + str(avg_num_trx_week)] = X_train.TransactionAmt.rolling(avg_num_trx_week, min_periods=avg_num_trx_day).min()
    X_train["rolling_max_" + str(avg_num_trx_week)] = X_train.TransactionAmt.rolling(avg_num_trx_week, min_periods=avg_num_trx_day).max()
    X_train["rolling_sum_" + str(avg_num_trx_week)] = X_train.TransactionAmt.rolling(avg_num_trx_week, min_periods=avg_num_trx_day).sum()

    X_train["rolling_min_diff_" + str(avg_num_trx_week)] = X_train.TransactionAmt.subtract(X_train["rolling_min_" + str(avg_num_trx_week)]).abs()
    X_train["rolling_max_diff_" + str(avg_num_trx_week)] = X_train.TransactionAmt.subtract(X_train["rolling_max_" + str(avg_num_trx_week)]).abs()
    X_train["rolling_sum_diff_" + str(avg_num_trx_week)] = X_train.TransactionAmt.subtract(X_train["rolling_sum_" + str(avg_num_trx_week)]).abs()
    
    return X_train
avg_num_trx_week = int(amount_with_less_99q.groupby(["Week"])["isFraud"].count().mean())
amount_with_less_99q = add_trx_rolling_features(amount_with_less_99q)
plt.figure(figsize=(15,5))
sns.boxenplot(x="rolling_max_diff_" + str(avg_num_trx_week), y="isFraud", 
             data=amount_with_less_99q)
plt.figure(figsize=(15,5))
sns.boxenplot(x="rolling_min_diff_" + str(avg_num_trx_week), y="isFraud", 
             data=amount_with_less_99q)
amount_with_less_99q[["rolling_min_diff_"+ str(avg_num_trx_week), "rolling_max_diff_"+ str(avg_num_trx_week)]].corr()
plt.figure(figsize=(15,5))
sns.boxenplot(x="rolling_sum_diff_" + str(avg_num_trx_week), y="isFraud", 
             data=amount_with_less_99q)
amount_with_less_99q[["offset_from_mean_week", "std_from_mean_week", "avg_trx_week", "std_trx_week"]].corr()
plot_category_trends(X_train, "ProductCD")
productCD = ["W", "H", "C", "S", "R"]
plt.figure(figsize=(15,5))
X_train["isFraud"] = X_train.isFraud.astype(bool)
sns.pointplot(x="ProductCD", y="TransactionAmt", data=X_train[X_train.isFraud], color="r", order=productCD)
sns.pointplot(x="ProductCD", y="TransactionAmt", data=X_train[~X_train.isFraud], color="b", order=productCD)
amount_with_less_99q["isFraud"] = amount_with_less_99q.isFraud.astype(bool)
plt.figure(figsize=(15,8))
sns.boxenplot(x="ProductCD", y="TransactionAmt", hue="isFraud", data=amount_with_less_99q, order=productCD)
X_train[["card1","card2","card3","card4","card5","card6"]].sample(5)
import category_encoders as ce
target_enc = ce.TargetEncoder()
card_encoded = target_enc.fit_transform(X_train[["card1","card2","card3","card4","card5","card6"]].astype("category"), X_train.isFraud.astype(bool))
card_encoded.columns = [col + "_encoded" for col in card_encoded.columns]
X_train = pd.concat([X_train, card_encoded], axis=1)
X_train.sample(5)
def plot_cum_curve(X_train, col, percent=0.2):
    percent_per_cat = (X_train[col].value_counts(ascending=True) / len(X_train[~X_train[col].isnull()])).to_frame()
    df_curve = percent_per_cat[col].expanding().sum().reset_index(drop=True).to_frame()
    
    percent_index = df_curve[df_curve[col] > percent].index[0] - 1
    
    percent2 =  np.round(percent_per_cat[col].tolist()[-1], 2)
    percent_index2 = df_curve[df_curve[col] > percent2].index[0] - 1
#     print(percent_per_cat.tail(1)[col])

    plt.figure(figsize=(15,4))
    label_str = str(percent_index) + " categories at " + str(percent*100) + "%"
    plt.axvline(percent_index, 0, 1, color="red", label=label_str)
    
    label_str2 = str(percent_index2) + " categories at " + str(percent2*100) + "%"
    plt.axvline(percent_index2, 0, 1, color="green", label=label_str2)
    
    ax = sns.lineplot(y=df_curve[col], x=df_curve.index)
    ax.set(title="Cumulative percentage curve " + col)
def reduce_category(X_train, col, percent=0.2):
    percent_per_cat = (X_train[col].value_counts(ascending=True) / len(X_train[~X_train[col].isnull()])).to_frame()
    df_curve = percent_per_cat[col].expanding().sum().to_frame().reset_index()
    
    percent_index = df_curve[df_curve[col] > percent_per_cat[col].tolist()[-1]].index[0] - 1
    
    minor_cat = df_curve.loc[:percent_index, "index"].tolist()
    
    X_train[col+"_reduce"] = X_train[col]
    max_value = int("9"*len(str(int(X_train[col].max()))))
    X_train.loc[X_train[col].isin(minor_cat), col+"_reduce"] = max_value
    
    return X_train
X_train["isFraud"] = X_train["isFraud"].astype("category")
for i in [1,2,3,5]:    
    cardstr = "card"+str(i)
    plot_cum_curve(X_train, cardstr)
    
    plt.figure(figsize=(15,4))
    sns.boxenplot(x=cardstr+"_encoded", y="isFraud", data=X_train)
X_train.addr1.astype("category").describe()
X_train.addr2.astype("category").describe()
target_enc2 = ce.TargetEncoder()
addr_encoded = target_enc2.fit_transform(X_train[["addr1","addr2"]].astype("category"), X_train.isFraud.astype(bool))
addr_encoded.columns = [col + "_encoded" for col in addr_encoded.columns]
X_train = pd.concat([X_train, addr_encoded], axis=1)
X_train.sample(5)
for i in [1,2]:
    addrstr = "addr"+str(i)
    plot_cum_curve(X_train, addrstr)
    plt.figure(figsize=(15,4))
    sns.boxenplot(x=addrstr+"_encoded", y="isFraud", data=X_train)
target_enc3 = ce.TargetEncoder()
email_encoded = target_enc3.fit_transform(X_train[["P_emaildomain","R_emaildomain"]].astype("category"), X_train.isFraud.astype(bool))
email_encoded.columns = [col + "_encoded" for col in email_encoded.columns]
X_train = pd.concat([X_train, email_encoded], axis=1)
X_train.sample(5)
for i in ["P","R"]:
    emailstr = str(i)+"_emaildomain"
    plot_cum_curve(X_train, emailstr)
    
    plt.figure(figsize=(15,4))
    sns.boxenplot(x=emailstr+"_encoded", y="isFraud", data=X_train)
X_train["isFraud"] = X_train["isFraud"].astype("category")
plt.figure(figsize=(15, 5))
sns.boxenplot(x="dist1", y="isFraud", data=X_train[X_train.dist1 < X_train.dist1.quantile(0.99)])
plt.figure(figsize=(15, 5))
sns.boxenplot(x="dist2", y="isFraud", data=X_train[X_train.dist2 < X_train.dist2.quantile(0.99)])
for feature in ['P_emaildomain','R_emaildomain']:
    plt.figure(figsize=(15, 20))
    domainlist = amount_with_less_99q[amount_with_less_99q.isFraud][feature].unique().tolist()
    sns.countplot(y=feature, x="TransactionAmt", data=amount_with_less_99q[amount_with_less_99q[feature].isin(domainlist)], hue="isFraud")
#     sns.boxenplot(y=feature, x="TransactionAmt", data=amount_with_less_99q[amount_with_less_99q.isFraud], color="r")
#     sns.boxenplot(y=feature, x="TransactionAmt", data=amount_with_less_99q[~amount_with_less_99q.isFraud & amount_with_less_99q[feature].isin(domainlist)], color="g")
for feature in ['card4','card6']:
    plt.figure(figsize=(15, 5))
    domainlist = amount_with_less_99q[amount_with_less_99q.isFraud][feature].unique().tolist()
    sns.boxenplot(x=feature, y="TransactionAmt", data=amount_with_less_99q[amount_with_less_99q[feature].isin(domainlist)], hue="isFraud")
plt.figure(figsize=(15,5))
sns.regplot(x="TransactionAmt", y="dist1", data=amount_with_less_99q[amount_with_less_99q.isFraud])
plt.figure(figsize=(15,5))
sns.regplot(x="TransactionAmt", y="dist2", data=amount_with_less_99q[amount_with_less_99q.isFraud])
X_train.sample(10)
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline, make_pipeline
CAT_STR_FEATURES = ["ProductCD", "P_emaildomain", "R_emaildomain", "card4", "card6"]
CAT_STR_FEATURES.extend(["M"+str(i+1) for i in range(9)])

CAT_NUM_FEATURES = ["addr1", "addr2"]
CAT_NUM_FEATURES.extend(["card"+str(i+1) for i in range(6) if (i!=3 and i!=5)])
CAT_FEATURES = CAT_STR_FEATURES + CAT_NUM_FEATURES

NUM_FEATURES = ["TransactionAmt"]
NUM_FEATURES.extend(["dist1", "dist2"])
NUM_FEATURES.extend(["C"+str(i+1) for i in range(14)])
NUM_FEATURES.extend(["D"+str(i+1) for i in range(15)])

ID_FEATURES = ["TransactionID", "TransactionDT"]
FLAG_FEATURES = ["with_identity"]
NON_VESTA_FEATURES = CAT_FEATURES + NUM_FEATURES + FLAG_FEATURES
# VESTA_FEATURES = list(set(X_train.drop(columns=["isFraud"]).columns) - set(NON_VESTA_FEATURES) - set(ID_FEATURES))
VESTA_FEATURES = list(set(X_transaction.drop(columns=["isFraud"]).columns) - set(NON_VESTA_FEATURES) - set(ID_FEATURES))
def add_day_count(X_train, min_transactiondt, day_of_week=True):
    seconds_per_day = 60*60*24
    seconds_per_week = seconds_per_day*7
    seconds_per_semimonthly = seconds_per_day*15
    seconds_per_month = seconds_per_day*30

    datetime_bin_day = np.arange(min_transactiondt, np.max(X_train["TransactionDT"])+seconds_per_day, seconds_per_day)
    datetime_bin_week = np.arange(min_transactiondt, np.max(X_train["TransactionDT"])+seconds_per_week, seconds_per_week)
    datetime_bin_semimonthly = np.arange(min_transactiondt, np.max(X_train["TransactionDT"])+seconds_per_semimonthly, seconds_per_semimonthly)
    datetime_bin_month = np.arange(min_transactiondt, np.max(X_train["TransactionDT"])+seconds_per_month, seconds_per_month)

    X_train["Day"] = pd.cut(X_train["TransactionDT"], bins=datetime_bin_day, labels=range(len(datetime_bin_day)-1)).astype(np.uint8).add(1)
    X_train["Week"] = pd.cut(X_train["TransactionDT"],  bins=datetime_bin_week, labels=range(len(datetime_bin_week)-1)).astype(np.uint8).add(1)
    
    if day_of_week:
        DAYS_IN_WEEK = 7 
        X_train["DayOfWeek"] = DAYS_IN_WEEK
        for i in range(1, DAYS_IN_WEEK):
            X_train.loc[(X_train.Day % DAYS_IN_WEEK) == i, "DayOfWeek"] = i

        WEEKS_IN_MONTH = 4 
        X_train["WeekOfMonth4"] = WEEKS_IN_MONTH
        for i in range(1, WEEKS_IN_MONTH):
            X_train.loc[(X_train.Week % WEEKS_IN_MONTH) == i, "WeekOfMonth4"] = i

        WEEKS_IN_MONTH = WEEKS_IN_MONTH + 1
        X_train["WeekOfMonth5"] = WEEKS_IN_MONTH
        for i in range(1, WEEKS_IN_MONTH):
            X_train.loc[(X_train.Week % WEEKS_IN_MONTH) == i, "WeekOfMonth5"] = i
    
    return X_train

def add_trx_features(X_train):
    avg_trx_week = X_train.groupby(["Week"]).TransactionAmt.mean().to_dict()
    X_train["avg_trx_week"] = X_train.Week.map(avg_trx_week).astype(np.float64)
    X_train["offset_from_mean_week"] = X_train.TransactionAmt.subtract(X_train.avg_trx_week).abs()
    X_train["offset_from_mean_week_2"] = X_train["offset_from_mean_week"].pow(2)

    std_trx_week = X_train.groupby(["Week"]).TransactionAmt.std().to_dict()
    X_train["std_trx_week"] = X_train.Week.map(std_trx_week).astype(np.float64)
    X_train["std_from_mean_week"] = X_train.TransactionAmt.divide(X_train.std_trx_week)

    return X_train

def add_trx_rolling_features(X_train, avg_num_trx_week, avg_num_trx_day):
    X_train["rolling_mean_" + str(avg_num_trx_week)] = X_train.TransactionAmt.rolling(avg_num_trx_week, min_periods=avg_num_trx_day).mean().fillna(-1)
    X_train["rolling_std_" + str(avg_num_trx_week)] = X_train.TransactionAmt.rolling(avg_num_trx_week, min_periods=avg_num_trx_day).std().fillna(-1)
    X_train["rolling_min_" + str(avg_num_trx_week)] = X_train.TransactionAmt.rolling(avg_num_trx_week, min_periods=avg_num_trx_day).min().fillna(0)
    X_train["rolling_max_" + str(avg_num_trx_week)] = X_train.TransactionAmt.rolling(avg_num_trx_week, min_periods=avg_num_trx_day).max().fillna(0)
    X_train["rolling_sum_" + str(avg_num_trx_week)] = X_train.TransactionAmt.rolling(avg_num_trx_week, min_periods=avg_num_trx_day).sum().fillna(0)

    X_train["rolling_min_diff_" + str(avg_num_trx_week)] = X_train.TransactionAmt.subtract(X_train["rolling_min_" + str(avg_num_trx_week)]).abs()
    X_train["rolling_max_diff_" + str(avg_num_trx_week)] = X_train.TransactionAmt.subtract(X_train["rolling_max_" + str(avg_num_trx_week)]).abs()
    X_train["rolling_sum_diff_" + str(avg_num_trx_week)] = X_train.TransactionAmt.subtract(X_train["rolling_sum_" + str(avg_num_trx_week)]).abs()
    
    return X_train
class TransactionDtTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.min_trx_dt = -1
    
    def fit(self, X, y=None):
        self.min_trx_dt = np.min(X["TransactionDT"])-1
        
        seconds_per_day = 60*60*24
        seconds_per_week = seconds_per_day*7
        seconds_per_semimonthly = seconds_per_day*15
        seconds_per_month = seconds_per_day*30

        datetime_bin_day = np.arange(self.min_trx_dt, np.max(X["TransactionDT"])+seconds_per_day, seconds_per_day)
        datetime_bin_week = np.arange(self.min_trx_dt, np.max(X["TransactionDT"])+seconds_per_week, seconds_per_week)
        datetime_bin_semimonthly = np.arange(self.min_trx_dt, np.max(X["TransactionDT"])+seconds_per_semimonthly, seconds_per_semimonthly)
        datetime_bin_month = np.arange(self.min_trx_dt, np.max(X["TransactionDT"])+seconds_per_month, seconds_per_month)

        X["Day"] = pd.cut(X["TransactionDT"], bins=datetime_bin_day, labels=range(len(datetime_bin_day)-1)).astype(np.uint8).add(1)
        X["Week"] = pd.cut(X["TransactionDT"],  bins=datetime_bin_week, labels=range(len(datetime_bin_week)-1)).astype(np.uint8).add(1)
    
        self.avg_num_trx_day = int(X.groupby(["Day"]).TransactionID.count().mean())
        self.avg_num_trx_week = int(X.groupby(["Week"]).TransactionID.count().mean())
        
        X = X.drop(columns=["Day", "Week"])
        
        return self
        
    def transform(self, X):
        X = add_day_count(X, self.min_trx_dt)
        X = add_trx_features(X)
        X = add_trx_rolling_features(X, 1000, 100)
        X = X.drop(columns=["Day", "Week"])
        return X
def compute_fraud_rate(X_train, y):
    rate_map = dict()
    for col in CAT_FEATURES:
        dtype = X_train[col].dtype
        X_train[col] = X_train[col].astype("category")
        
        fraud_count = X_train[y.astype(bool)].groupby([col]).TransactionID.count()
        fraud_rate = (fraud_count / X_train.groupby([col]).TransactionID.count()).to_frame()
        contribution_rate = (fraud_count / y.astype(bool).sum()).to_frame()
        
        fraud_rate["TransactionID"] = fraud_rate["TransactionID"] / np.linalg.norm(fraud_rate["TransactionID"], ord=1)
        
        rate_map[col] = [fraud_rate.fillna(0).TransactionID.to_dict(), 
                         contribution_rate.fillna(0).TransactionID.to_dict()]
        X_train[col] = X_train[col].astype(dtype)
        
    return rate_map
def add_amt_features(X_train, rate_map):
    X_train["TransactionAmt2"] = X_train["TransactionAmt"].pow(2)
    
    for colkey in rate_map.keys():
        col_name = "TransactionAmt_"+colkey
        fraud_rate = rate_map[colkey][0]
        contribution_rate = rate_map[colkey][1]
        
        X_train[col_name] = X_train[colkey].map(fraud_rate).multiply(X_train.TransactionAmt).pow(2)
        X_train[col_name] = X_train[col_name].add(X_train[colkey].map(contribution_rate).multiply(X_train.TransactionAmt)).fillna(0)
        
    return X_train
class TransactionAmtTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.fraud_rate = dict()
        self.min_trx_dt = -1
    
    def fit(self, X, y=None):
        self.fraud_rate = compute_fraud_rate(X, y)
        return self
        
    def transform(self, X):
        X = add_amt_features(X, self.fraud_rate)
        return X
def reduce_category(X_train, col, cat_to_replace, same_count_topn):
    percent_per_cat = (X_train[col].value_counts(ascending=True) / len(X_train[~X_train[col].isnull()])).to_frame()
    df_curve = percent_per_cat[col].expanding().sum().to_frame().reset_index()
    
    percent_index = df_curve[df_curve[col] > percent_per_cat[col].tolist()[-same_count_topn]].index[0] - 1
    
    minor_cat = df_curve.loc[:percent_index, "index"].tolist()
    X_train.loc[X_train[col].isin(minor_cat), col] = cat_to_replace
    
    return X_train, minor_cat
def agg_category(X_train, col, category_left=15, same_count_topn=1):
    i = 0
    minor_cat_dict = dict()
    total_categories = X_train[col].nunique()
    
    while total_categories > category_left + i:
        max_value = int("9"*len(str(int(X_train[col].max())))) - i
        X_train, minor_cat = reduce_category(X_train, col, max_value, same_count_topn)
        
        total_categories = X_train[col].nunique()
        for cat in minor_cat:
            minor_cat_dict[cat] = max_value
        i = i + 1
        
    return minor_cat_dict
class CategoryReducerTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.rare_cat = dict()
    
    def fit(self, X, y=None):
        for col in X.columns:
            minor_cat_dict = agg_category(X.copy(), col)
            self.rare_cat[col] = minor_cat_dict
        return self
    
    def transform(self, X):
        for col in X.columns: 
            X.loc[X[col].isin(self.rare_cat[col].keys()),col] = X[col].map(self.rare_cat[col])
        return X
import category_encoders as ce
class LeaveOneOutTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.loo_encoder = None
        self.bin_encoder = None
        
    def fit(self, X, y=None):
        self.loo_encoder = ce.LeaveOneOutEncoder(cols=X.columns)
        self.loo_encoder.fit(X, y)
        
        self.bin_encoder = ce.TargetEncoder(cols=X.columns)
        self.bin_encoder.fit(X, y)
        return self
    
    def transform(self, X):
        existing_cols = X.columns
        X[[col + "_loo" for col in X.columns]] = self.loo_encoder.transform(X)
        X = pd.concat([X, self.bin_encoder.transform(X[existing_cols])], axis=1) 
        X = X.drop(columns=existing_cols)
        return X
class WeightOfEvidenceTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.woe_encoder = None
        
    def fit(self, X, y=None):
        self.woe_encoder = ce.WOEEncoder(cols=X.columns)
        self.woe_encoder.fit(X, y)
        return self
    
    def transform(self, X):
        X[[col + "_woe" for col in X.columns]] = self.woe_encoder.transform(X)
        return X
class CategoryTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        for col in X.columns:
            X[col] = X[col].astype("Int64").astype(str).str.split('.', expand=True)[0]
        return X
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

class IterativeImputerTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, n_nearest_features, max_iter):
        self.n_nearest_features = n_nearest_features
        self.max_iter = max_iter
        self.iterative_imputer = None
    
    def fit(self, X, y=None):
        self.iterative_imputer = IterativeImputer(random_state=1, 
                                                  n_nearest_features=self.n_nearest_features, 
                                                  max_iter=self.max_iter, skip_complete=True)
        self.iterative_imputer.fit(X)
        return self
    
    def transform(self, X):
        return pd.DataFrame(self.iterative_imputer.transform(X), columns=X.columns, index=X.index)
from sklearn.impute import SimpleImputer
class SimpleImputerTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, strategy, fill_value):
        self.strategy = strategy
        self.fill_value = fill_value
        self.simple_imputer = None
    
    def fit(self, X, y=None):
        self.simple_imputer = SimpleImputer(strategy=self.strategy, fill_value=self.fill_value)
        self.simple_imputer.fit(X)
        return self
    
    def transform(self, X):
        return pd.DataFrame(self.simple_imputer.transform(X), columns=X.columns, index=X.index)
from sklearn.impute import MissingIndicator
class MissingIndicatorTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.columns = None
        self.missing_indicator = None
    
    def fit(self, X, y=None):
        self.columns = X.columns
        self.missing_indicator = MissingIndicator()
        self.missing_indicator.fit(X)
        return self
    
    def transform(self, X):
        return pd.DataFrame(self.missing_indicator.transform(X), columns=X.columns)
# Use FeatureUnion instead with ColumnSelector, column names are getting swapped
class DfColumnTransformer(TransformerMixin, BaseEstimator):
    def __init__(self,
                 transformers, *,
                 remainder='drop',
                 sparse_threshold=0.3,
                 n_jobs=None,
                 transformer_weights=None,
                 verbose=False):
        self.col_transformer = ColumnTransformer(
            transformers=transformers,
            remainder=remainder,
            sparse_threshold=sparse_threshold,
            n_jobs=n_jobs,
            transformer_weights=transformer_weights,
            verbose=verbose)
        
        params = self.col_transformer.get_params()
        self.remainder = params["remainder"]    
        self.index = None
        self.columns = list()
    
        for tf in params["transformers"]:
            self.columns.extend(tf[2])
    
    def fit(self, X, y=None):
        self.index = X.index
        self.columns = X.columns if self.remainder=="passthrough" else self.columns
        self.col_transformer.fit(X)
        return self
    
    def transform(self, X):
        return pd.DataFrame(self.col_transformer.transform(X), columns=self.columns, index=self.index)
class ColumnSelector(TransformerMixin, BaseEstimator):
    def __init__(self, columns, inverse=False):
        self.inverse = inverse
        self.columns = columns
        self.receieved_cols = None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[list(set(X.columns) - set(self.columns)) if self.inverse else self.columns]
from sklearn.externals.joblib import Parallel, delayed
from sklearn.pipeline import FeatureUnion, _fit_transform_one, _transform_one, _name_estimators
from scipy import sparse

class FeatureUnionDf(FeatureUnion):
    def fit_transform(self, X, y=None, **fit_params):
        results = self._parallel_func(X, y, fit_params, _fit_transform_one)
        if not results:
            return np.zeros((X.shape[0], 0))

        Xs, transformers = zip(*results)
        self._update_transformer_list(transformers)

        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self._merge_dataframe(Xs)
        return Xs
    
    def transform(self, X):
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, X, None, weight)
            for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self._merge_dataframe(Xs)
        print(Xs.shape)
        return Xs
    
    def _merge_dataframe(self, X):
        return pd.concat(X, axis="columns", copy=False)
     
def make_uniondf(*transformers, **kwargs):
    n_jobs = kwargs.pop('n_jobs', None)
    verbose = kwargs.pop('verbose', False)
    return FeatureUnionDf(_name_estimators(transformers), n_jobs=n_jobs, verbose=verbose)
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, mutual_info_classif, f_classif
class FeatureSelectorTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, cat_score_func, n_features):
        self.n_features = n_features
        self.cat_score_func = cat_score_func
        self.kbest_num = None
        self.kbest_cat = None
        self.selected_num_cols = None
        self.selected_cat_cols = None
        self.num_columns = None
        self.cat_columns = None
        
    def fit(self, X, y=None):
        self.cat_columns = CAT_FEATURES + ["DayOfWeek", "WeekOfMonth4", "WeekOfMonth5"]
        self.num_columns = list(set(X.drop(columns=["TransactionID", "TransactionDT"]).columns) - set(self.cat_columns))
        
        self.kbest_num = SelectKBest(score_func=f_classif, k=int(self.n_features*0.9))
        self.kbest_num.fit(X[self.num_columns], y)
        
        self.kbest_cat = SelectKBest(score_func=self.cat_score_func, k=int(self.n_features*0.1))
        self.kbest_cat.fit(X[self.cat_columns], y)
        
        return self
        
    def transform(self, X):
        self.kbest_num.transform(X[self.num_columns])
        self.kbest_cat.transform(X[self.cat_columns])
        self.selected_num_cols = X[self.num_columns].loc[:,self.kbest_num.get_support()].columns.tolist()
        self.selected_cat_cols = X[self.cat_columns].loc[:,self.kbest_cat.get_support()].columns.tolist()
        
        return X.loc[:, self.selected_num_cols+self.selected_cat_cols]
#DONT FORGET TO DROP TRANSACTION IDS AND DTS
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
class FeatureSelectorTransformer2(TransformerMixin, BaseEstimator):
    def __init__(self, model, n_features, threshold):
        self.model = model
        self.threshold = threshold
        self.n_features = n_features
        self.feature_importances = None
        self.select_from_model = None
        
    def fit(self, X, y=None):
        self.all_columns = X.columns
        self.select_from_model = SelectFromModel(self.model, max_features=self.n_features, threshold=self.threshold)
        self.select_from_model.fit(X.drop(columns=["TransactionID", "TransactionDT"]), y)
        return self
    
    def transform(self, X):
        self.feature_importances = sorted(list(zip(X.columns, self.select_from_model.estimator_.feature_importances_)), key=lambda x: x[1], reverse=True) 
        return X.drop(columns=["TransactionID", "TransactionDT"]).loc[:,self.select_from_model.get_support()]
from sklearn.preprocessing import StandardScaler, MinMaxScaler
class StandardScalerTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.scaler = None
        
    def fit(self, X, y=None):
        self.scaler = MinMaxScaler()
        self.scaler.fit(X)
        return self
    
    def transform(self, X):
        return pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)
from sklearn.preprocessing import PolynomialFeatures
class PolynomialTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.poly = None
        
    def fit(self, X, y=None):
        self.poly = PolynomialFeatures(interaction_only=True, include_bias=False)
        return self
    
    def transform(self, X):
        dfpoly = self.poly.fit_transform(X)
        dfpoly = pd.DataFrame(dfpoly[:,len(X.columns):], columns=["poly"+str(i) for i in range(dfpoly.shape[1]-len(X.columns))], index=X.index).astype(np.float32)
        return pd.concat([X, dfpoly], axis=1)
from sklearn.pipeline import FeatureUnion, make_union
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, mutual_info_classif, f_classif

def init_pipeline(X_train):
    X_train_sample = X_train
    # X_train_sample = pd.concat([X_train, X_valid]).reset_index(drop=True)

    for col in ["TransactionAmt"]:
        if not np.isnan(X_train_sample[col].quantile(0.99)):
            X_train_sample = X_train_sample.loc[X_train_sample[col] < X_train_sample[col].quantile(0.99), :]

    y = X_train_sample.isFraud.astype(int)
    X_train_sample = X_train_sample.drop(columns=["isFraud"])

    class_weight = compute_class_weight("balanced", np.unique(y.to_numpy()), y.to_numpy())
    class_weight = dict(zip(range(y.nunique()), class_weight))
    feature_model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=1, class_weight=class_weight, verbose=4, n_jobs=-1)

    pipeline = make_pipeline(
        make_uniondf(
            make_pipeline(
                ColumnSelector(CAT_NUM_FEATURES),
                CategoryReducerTransformer(),
                CategoryTransformer(),
                WeightOfEvidenceTransformer(),
                LeaveOneOutTransformer(),
                verbose=True
            ),
            make_pipeline(
                ColumnSelector(CAT_STR_FEATURES),
                WeightOfEvidenceTransformer(),
                LeaveOneOutTransformer(),
                verbose=True
            ),
            make_pipeline(
                ColumnSelector(VESTA_FEATURES + NUM_FEATURES),
                SimpleImputerTransformer(strategy="constant", fill_value=-1),
                verbose=True
            ),
            make_pipeline(ColumnSelector(CAT_NUM_FEATURES + CAT_STR_FEATURES + VESTA_FEATURES + NUM_FEATURES,
                                        inverse=True)),
            verbose=True
        ),
        make_uniondf(
            make_pipeline(
                ColumnSelector(VESTA_FEATURES, inverse=True),
                IterativeImputerTransformer(n_nearest_features=10, max_iter=20),
                verbose=True
            ),
            make_pipeline(ColumnSelector(VESTA_FEATURES)),
            verbose=True
        ),
        make_uniondf(
            make_pipeline(
                ColumnSelector(["C"+str(i+1) for i in range(14)] + ["TransactionAmt"]),
                PolynomialTransformer()
            ),
            make_pipeline(ColumnSelector(["C"+str(i+1) for i in range(14)] + ["TransactionAmt"], inverse=True)),
            verbose=True
        ),
#         TransactionDtTransformer(),
#         TransactionAmtTransformer(),
    #     make_uniondf(
    #         make_pipeline(
    #             ColumnSelector(CAT_FEATURES + FLAG_FEATURES + VESTA_FEATURES + ["DayOfWeek", "WeekOfMonth4", "WeekOfMonth5"],
    #                           inverse=True),
    #             StandardScalerTransformer()
    #         ),
    #         make_pipeline(
    #             ColumnSelector(CAT_FEATURES + FLAG_FEATURES + VESTA_FEATURES + ["DayOfWeek", "WeekOfMonth4", "WeekOfMonth5"]))
    #     ),
    #     FeatureSelectorTransformer(mutual_info_classif, n_features=200),
        FeatureSelectorTransformer2(feature_model, n_features=350, threshold=-np.inf),
    #     ColumnSelector(VESTA_FEATURES, inverse=True),
        verbose=True
    )
    
    return pipeline, X_train_sample, y
def execute_pipeline(X_train, X_valid):    
    pipeline, X_train_sample, y = init_pipeline(X_train)
    X_train_sample = pipeline.fit_transform(X_train_sample, y)
    
    X_valid2 = pipeline.transform(X_valid.drop(columns=["isFraud"]))
    y_valid = X_valid.isFraud.astype(int)
    
    class_weight = compute_class_weight("balanced", np.unique(y.to_numpy()), y.to_numpy())
    cboost2 = CatBoostClassifier(loss_function="Logloss", random_seed=50, class_weights=class_weight, iterations=1000)
    cboost2.fit(X_train_sample.to_numpy(), y,
                eval_set=catboost.Pool(X_valid2.to_numpy(), label=y_valid), 
                plot=True, early_stopping_rounds=100)
    
    from sklearn.metrics import classification_report
    print(classification_report(y_valid, cboost2.predict(X_valid2).reshape(-1)))
    from sklearn.metrics import roc_auc_score
    print(roc_auc_score(y_valid, cboost2.predict(X_valid2).reshape(-1), average="weighted"))
    
    # X_test2 = pipeline.transform(X_test.drop(columns=["isFraud"]).copy())
    # y_test = X_test.isFraud.astype(int)
    
    print("Number of features", len(X_train_sample.columns))
    print("Features", X_train_sample.columns)
    
    del X_valid, X_valid2, cboost2
    gc.collect()
    
    return X_train_sample
X_train_sample = execute_pipeline(X_train.copy(), X_valid.copy())
#               precision    recall  f1-score   support

#            0       0.99      0.91      0.95    106379
#            1       0.24      0.72      0.36      4348

#     accuracy                           0.90    110727
#    macro avg       0.62      0.81      0.66    110727
# weighted avg       0.96      0.90      0.92    110727

# 0.8148787640462721
#               precision    recall  f1-score   support

#            0       0.99      0.88      0.93    106379
#            1       0.21      0.76      0.32      4348

#     accuracy                           0.88    110727
#    macro avg       0.60      0.82      0.63    110727
# weighted avg       0.96      0.88      0.91    110727

# 0.8198649641226113
#               precision    recall  f1-score   support
#            0       0.99      0.93      0.96    106379
#            1       0.29      0.70      0.41      4348

#     accuracy                           0.92    110727
#    macro avg       0.64      0.81      0.69    110727
# weighted avg       0.96      0.92      0.94    110727

# ROC-AUC score: 0.8141551715515302
pipeline, X_train_sample, y = init_pipeline(X_transaction.reset_index(drop=True))
X_train_sample = pipeline.fit_transform(X_train_sample, y)

# X_valid2 = pipeline.transform(X_valid.drop(columns=["isFraud"]))
# y_valid = X_valid.isFraud.astype(int)

class_weight = compute_class_weight("balanced", np.unique(y.to_numpy()), y.to_numpy())
cboost = CatBoostClassifier(loss_function="Logloss", random_seed=50, class_weights=class_weight, iterations=180)
cboost.fit(X_train_sample.to_numpy(), y,
            plot=True, early_stopping_rounds=100)
test = pd.read_csv("../input/ieee-fraud-detection/test_transaction.csv")
identity = pd.read_csv("../input/ieee-fraud-detection/test_identity.csv", usecols=["TransactionID"])
id_with_identity = identity["TransactionID"].unique().tolist()
test["with_identity"] = test["TransactionID"].isin(id_with_identity)

ids = test.TransactionID
test = pipeline.transform(test)
submission = pd.concat([ids, pd.Series(cboost.predict(test).reshape(-1))])
submission.to_csv("submission.csv", index=False)