!pip install fairlearn
import fairlearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightgbm as lgb

# from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from imblearn.ensemble import BalancedBaggingClassifier

from fairlearn.widget import FairlearnDashboard
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.metrics import (
    group_summary, selection_rate, selection_rate_group_summary,
    demographic_parity_difference, demographic_parity_ratio,
    balanced_accuracy_score_group_summary, roc_auc_score_group_summary,
    equalized_odds_difference, difference_from_summary)

# Helper functions
def get_metrics_df(models_dict, y_true, group):
    metrics_dict = {
        "Overall selection rate": (
            lambda x: selection_rate(y_true, x), True),
        "Demographic parity difference": (
            lambda x: demographic_parity_difference(y_true, x, sensitive_features=group), True),
        "Demographic parity ratio": (
            lambda x: demographic_parity_ratio(y_true, x, sensitive_features=group), True),
        "-----": (lambda x: "", True),
        "Overall balanced error rate": (
            lambda x: 1-balanced_accuracy_score(y_true, x), True),
        "Balanced error rate difference": (
            lambda x: difference_from_summary(
                balanced_accuracy_score_group_summary(y_true, x, sensitive_features=group)), True),
        "Equalized odds difference": (
            lambda x: equalized_odds_difference(y_true, x, sensitive_features=group), True),
        "------": (lambda x: "", True),
        "Overall AUC": (
            lambda x: roc_auc_score(y_true, x), False),
        "AUC difference": (
            lambda x: difference_from_summary(
                roc_auc_score_group_summary(y_true, x, sensitive_features=group)), False),
    }
    df_dict = {}
    for metric_name, (metric_func, use_preds) in metrics_dict.items():
        df_dict[metric_name] = [metric_func(preds) if use_preds else metric_func(scores) 
                                for model_name, (preds, scores) in models_dict.items()]
    return pd.DataFrame.from_dict(df_dict, orient="index", columns=models_dict.keys())
df_orig = pd.read_csv('/kaggle/input/promotion.csv')
df_orig.head()
df = df_orig.copy()
df.dtypes
obj_columns = df.select_dtypes(['object']).columns
for c in obj_columns:
    df[c] = df[c].astype('category')

cat_columns = df.select_dtypes(['category']).columns
df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

df.dtypes
df.isnull().sum()
df['education'] = df['education'].fillna(df['education'].mode()[0])
df['previous_year_rating'] = df['previous_year_rating'].fillna(df['previous_year_rating'].mode()[0])

print("Number of missing values:", df.isnull().sum().sum())
plt.rcParams['figure.figsize'] = (15, 5)
plt.style.use('seaborn-white')

df['is_promoted'].value_counts().plot(kind = 'pie',
                                      autopct = '%.2f%%',
                                      startangle = 90,
                                      labels = ['Not promoted','Promoted'],
                                      pctdistance = 0.5)
plt.axis('off')

plt.suptitle('Target Class Balance', fontsize = 16)
plt.show()
# Creating a label vector and a feature vector
X = df.drop(["employee_id","is_promoted"], axis=1)
y = df["is_promoted"]

A = df_orig['department']

X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(X, 
                                                                     y, 
                                                                     A,
                                                                     test_size = 0.2,
                                                                     random_state=0,
                                                                     stratify=y)
plt.rcParams['figure.figsize'] = (15, 5)
plt.style.use('seaborn-white')

plt.subplot(1, 2, 1)

y_train.value_counts().plot(kind = 'pie',
                            autopct = '%.2f%%',
                            startangle = 90,
                            labels = ['Not promoted','Promoted'],
                            pctdistance = 0.5)

plt.xlabel('Training dataset', fontsize = 14)

plt.subplot(1, 2, 2)

y_test.value_counts().plot(kind = 'pie',
                           autopct = '%.2f%%',
                           startangle = 90,
                           labels = ['Not promoted','Promoted'],
                           pctdistance = 0.5)

plt.xlabel('Testing dataset', fontsize = 14)

plt.suptitle('Target Class Balance', fontsize = 16)
plt.show()
clf = BalancedBaggingClassifier()
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(clf, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
print('Mean ROC AUC: %.3f' % np.mean(scores))
clf.fit(X_train, y_train)

test_scores = clf.predict_proba(X_test)[:, 1]

# Predictions (0 or 1) on test set
y_pred = (test_scores >= np.mean(y_test)) * 1

print('Mean ROC AUC: %.3f' % roc_auc_score(y_test, test_scores))
importances = np.mean([est.steps[1][1].feature_importances_ for est in clf.estimators_], axis=0)

indices = np.argsort(importances)[::-1]

# Plot the impurity-based feature importances of the forest
plt.figure()
plt.style.use('seaborn-white')
plt.title("Feature importances", fontsize = 16)
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns, rotation='vertical')
plt.xlim([-1, X.shape[1]])
plt.show()
models_dict = {"Unmitigated": (y_pred, test_scores)}
get_metrics_df(models_dict, y_test, A_test)
gs = group_summary(roc_auc_score, y_test, y_pred, sensitive_features=A_test)
gs
plt.figure()
plt.style.use('seaborn-white')
plt.title("AUC per group before mitigating model biases", fontsize = 16)
plt.bar(range(len(gs["by_group"])), list(gs["by_group"].values()), align='center')
plt.xticks(range(len(gs["by_group"])), list(gs["by_group"].keys()))
plt.ylim(0, 1)
plt.show()
srg = selection_rate_group_summary(y_test, y_pred, sensitive_features=A_test)

plt.figure()
plt.style.use('seaborn-white')
plt.title("Selection rate per group before mitigating model biases", fontsize = 16)
plt.bar(range(len(srg["by_group"])), list(srg["by_group"].values()), align='center')
plt.xticks(range(len(srg["by_group"])), list(srg["by_group"].keys()))
plt.ylim(0, 1)
plt.show()
FairlearnDashboard(sensitive_features=A_test, sensitive_feature_names=['department'],
                   y_true=y_test,
                   y_pred={"Unmitigated": y_pred})
lgb_params = {
    'objective' : 'binary',
    'metric' : 'auc',
    'learning_rate': 0.03,
    'num_leaves' : 10,
    'max_depth' : 3,
    'random_state': 9
}

np.random.seed(0)  # set seed for consistent results with ExponentiatedGradient

constraint = DemographicParity()
clf = lgb.LGBMClassifier(**lgb_params)
mitigator = ExponentiatedGradient(clf, constraint)
mitigator.fit(X_train, y_train, sensitive_features=A_train)

y_pred_mitigated = mitigator.predict(X_test)
models_dict = {"ExponentiatedGradient": (y_pred_mitigated, y_pred_mitigated)}
get_metrics_df(models_dict, y_test, A_test)
gs = group_summary(roc_auc_score, y_test, y_pred_mitigated, sensitive_features=A_test)
gs
plt.figure()
plt.style.use('seaborn-white')
plt.title("Group summary after mitigating model biases", fontsize = 16)
plt.bar(range(len(gs["by_group"])), list(gs["by_group"].values()), align='center')
plt.xticks(range(len(gs["by_group"])), list(gs["by_group"].keys()))
plt.ylim(0, 1)
plt.show()
srg = selection_rate_group_summary(y_test, y_pred_mitigated, sensitive_features=A_test)

plt.figure()
plt.style.use('seaborn-white')
plt.title("Selection rate per group after mitigating model biases", fontsize = 16)
plt.bar(range(len(srg["by_group"])), list(srg["by_group"].values()), align='center')
plt.xticks(range(len(srg["by_group"])), list(srg["by_group"].keys()))
plt.ylim(0, 1)
plt.show()
# The widget does not show up in Kaggle Kernels (see examples here: https://fairlearn.github.io/quickstart.html)
FairlearnDashboard(sensitive_features=A_test, sensitive_feature_names=['department'],
                   y_true=y_test,
                   y_pred={"Unmitigated": y_pred,
                           "ExponentiatedGradient": y_pred_mitigated})