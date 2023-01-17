import numpy as np

import pandas as pd

import lightgbm as lgb

import optuna

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_recall_curve, average_precision_score, plot_confusion_matrix 

import matplotlib.pyplot as plt

import seaborn as sns

import os



%matplotlib inline

sns.set_context("talk")

sns.set(rc={"figure.figsize":[16,9]})



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")

print(df.shape)

df.head()
df.dtypes
for column in df.columns:

    nulls = pd.isnull(df[column]).sum()

    if nulls:

        print(column, pd.isnull(df[column]).sum())
PCA_features = [f"V{i}" for i in range(1,29)]

# PCA_descriptions = df.describe()[PCA_features]



plotdf = df[PCA_features].melt(ignore_index=False)
sns.violinplot(x="variable", y="value", data=plotdf)

plt.show()
df["log_amount"] = np.log(df["Amount"] + 1)
X = df[PCA_features + ["log_amount"]]

y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=99)



X_std = X_train.std()

X_train = X_train / X_std

X_test = X_test / X_std
clf = LogisticRegression()



clf.fit(X_train, y_train)

logreg_y_pred_proba = clf.predict_proba(X_test)[:,1]

logreg_y_pred = (logreg_y_pred_proba > 0.5).astype(int)



naive_model = np.ones(y_test.shape[0]) * y_train.mean()



PR_AUC = average_precision_score(y_test, logreg_y_pred_proba)

PR_curve = precision_recall_curve(y_test, logreg_y_pred_proba)

baseline_PR_AUC = average_precision_score(y_test, naive_model)

baseline_PR_curve = precision_recall_curve(y_test, naive_model)





print(f"baseline PR-AUC score for precision recall, predicting no fraud = {baseline_PR_AUC}")

print(f"PR-AUC score for basic logistic regression model = {PR_AUC}")
print("Each row sums to 100%:")

fig, ax = plt.subplots()

plot_confusion_matrix(clf, X_test, y_test, cmap=plt.cm.Blues, ax=ax, normalize="true")

ax.grid(False)

plt.show()
fig, ax = plt.subplots()

plt.plot(PR_curve[0], PR_curve[1])

plt.plot(baseline_PR_curve[0], baseline_PR_curve[1])

ax.set_xlabel("precision")

ax.set_ylabel("recall")

ax.legend(["Logistic Regression","Naive model"])

plt.show()
feature_importances = pd.DataFrame(data={

    "feature":PCA_features + ["log_amount"],

    "logreg_coefs": clf.coef_[0]

})



feature_importances["logreg_coefs_abs"] = feature_importances.logreg_coefs

feature_importances["logreg_rank"] = feature_importances.logreg_coefs.rank(ascending=False)
rf_clf = RandomForestClassifier()

rf_clf.fit(X_train, y_train)



rf_y_pred_proba = rf_clf.predict_proba(X_test)[:, 1]

rf_y_pred = (rf_y_pred_proba > 0.5).astype(int)



PR_AUC_rf = average_precision_score(y_test, rf_y_pred_proba)

PR_curve_rf = precision_recall_curve(y_test, rf_y_pred_proba)





print(f"baseline score for precision recall, predicting no fraud = {baseline_PR_AUC}")

print(f"score for basic random forest classification model = {PR_AUC_rf}")
fig,ax = plt.subplots()

plot_confusion_matrix(rf_clf, X_test, y_test, cmap=plt.cm.Blues, ax=ax, normalize="true")

ax.grid(False)

plt.show()
fig, ax = plt.subplots()

plt.plot(PR_curve_rf[0], PR_curve_rf[1])

plt.plot(PR_curve[0], PR_curve[1])

plt.plot(baseline_PR_curve[0], baseline_PR_curve[1])

ax.set_xlabel("precision")

ax.set_ylabel("recall")

ax.legend(["Random Forest", "Logistic Regression", "Naive model"])

plt.show()
feature_importances["randfor_importance"] = rf_clf.feature_importances_

feature_importances["randfor_rank"] = feature_importances["randfor_importance"].rank(ascending=False)
FIX_PARAMS = {

    'objective': 'binary',

    'metric': 'average_precision',

    'is_unbalance':True,

    'bagging_freq':5,

    'boosting':'dart',

}



def objective(trial):

    

    OPTIMIZE_PARAMS = {

        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.6, log=True),

         'max_depth': trial.suggest_int("max_depth", 1, 30), 

         'num_leaves': trial.suggest_int("num_leaves", 10, 200),

         'feature_fraction': trial.suggest_float("feature_fraction", 0.1, 1),

         'subsample': trial.suggest_float("subsample", 0.1, 1.0),

    }



    train_data = lgb.Dataset(X_train, label=y_train)

    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)



    params = {**FIX_PARAMS, **OPTIMIZE_PARAMS}



    model = lgb.train(params, train_data)

    y_pred = model.predict(X_test)

    PR_AUC_lgb = average_precision_score(y_test, y_pred)

    return PR_AUC_lgb



study = optuna.create_study(direction="maximize")

study.optimize(objective, n_trials=100)

trial = study.best_trial
# best lgb model

best_params = {**FIX_PARAMS,**{i[0]:i[1] for i in trial.params.items()}}



train_data = lgb.Dataset(X_train, label=y_train)

model = lgb.train(best_params, train_data)
y_pred = model.predict(X_test)



PR_AUC_lgb = average_precision_score(y_test, y_pred)

PR_curve_lgb = precision_recall_curve(y_test, y_pred)





print(f"baseline score for precision recall, predicting no fraud = {baseline_PR_AUC}")

print(f"score for lightgbm classification model = {PR_AUC_lgb}")



fig, ax = plt.subplots()

plt.plot(PR_curve_lgb[0], PR_curve_lgb[1])

plt.plot(PR_curve_rf[0], PR_curve_rf[1])

plt.plot(PR_curve[0], PR_curve[1])

plt.plot(baseline_PR_curve[0], baseline_PR_curve[1])

ax.set_xlabel("precision")

ax.set_ylabel("recall")

ax.legend([

    f"Lightgbm AUC: {PR_AUC_lgb:.2f}",

    f"Random Forest AUC: {PR_AUC_rf:.2f}",

    f"Logistic Regression AUC: {PR_AUC:.2f}",

    f"Naive model AUC: {baseline_PR_AUC:.2f}"

])

plt.show()
feature_importances["lgb_gain"] = model.feature_importance(importance_type="gain")

feature_importances["lgb_splits"] = model.feature_importance(importance_type="split")



feature_importances["lgb_gain_rank"] = feature_importances["lgb_gain"].rank(ascending=False)

feature_importances["lgb_split_rank"] = feature_importances["lgb_splits"].rank(ascending=False)

feature_importances["lgb_rank"] = feature_importances[["lgb_gain_rank", "lgb_split_rank"]].mean(axis=1).rank()



ranks = ["logreg_rank", "randfor_rank", "lgb_rank"]

feature_importances["mean_rank"] = feature_importances[ranks].mean(axis=1).rank()
feature_importances.sort_values(by="mean_rank").head(10)