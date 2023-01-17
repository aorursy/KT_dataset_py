import numpy as np

import pandas as pd

import os

import datetime as dt

import seaborn as sns

from matplotlib import pyplot as plt



csv_paths = []



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        csv_paths.append(os.path.join(dirname, filename))
print(csv_paths)
raw_dataframes = []

for path in csv_paths:

    df = pd.read_csv(path, index_col=0)

    raw_dataframes.append(df)
raw_dataframes[0].head()
raw_dataframes[1].head()
raw_hard_churn = raw_dataframes[0].copy()

raw_hard_churn.info()

raw_hard_churn.isna().sum()
raw_soft_churn = raw_dataframes[1].copy()

raw_soft_churn.info()

raw_soft_churn.isna().sum()
raw_soft_churn = raw_soft_churn[~raw_soft_churn.isnull().any(axis=1)]
raw_hard_churn.corr()
plt.figure(figsize=(14, 7))

plt.title("Feature Correlation (Hard)")

sns.heatmap(raw_hard_churn.loc[:, raw_hard_churn.columns != 'imei_name'].corr(), annot=True)
plt.figure(figsize=(14, 7))

plt.title("Feature Correlation (Soft)")

sns.heatmap(raw_soft_churn.loc[:, raw_soft_churn.columns != 'imei_name'].corr(), annot=True)
from sklearn.model_selection import train_test_split



# Shuffle is disabled and random state here is set for reproducibility



def split_data(dataframe, result_column, test_size):

    X, y = dataframe.loc[:, (dataframe.columns != result_column) & (dataframe.columns != 'imei_name')], dataframe[result_column]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=0, shuffle=False)

    return X_train, X_valid, y_train, y_valid



X_hard_train, X_hard_valid, y_hard_train, y_hard_valid = split_data(raw_hard_churn, 'churn', 0.3)

X_soft_train, X_soft_valid, y_soft_train, y_soft_valid = split_data(raw_soft_churn, 'churn', 0.3)
X_hard_train.head()
y_hard_train.head()
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import cross_validate

# Random state here is set for reproducibility



hard_models = {

    'RandomForestClassifier': {

        'model': RandomForestClassifier,

        'param': {

            'n_estimators': 155,

            'criterion': 'gini',

            'min_samples_leaf': 10,

            'max_features': 'sqrt',

            'random_state': 0

        }

    },

    'ExtraTreesClassifier': {

        'model': ExtraTreesClassifier,

        'param': {

            'n_estimators': 143,

            'criterion': 'gini',

            'min_samples_leaf': 10,

            'max_features': 'sqrt',

            'random_state': 0

        }

    },

    'AdaBoostClassifier': {

        'model': AdaBoostClassifier,

        'param': {

            'n_estimators': 170,

            'learning_rate': 0.14046548511432905,

            'random_state': 0

        }

    },

    'GradientBoostingClassifier': {

        'model': GradientBoostingClassifier,

        'param': {

            'n_estimators': 161,

            'learning_rate': 0.021480690032076512,

            'subsample': 1.0,

            'min_samples_leaf': 4,

            'max_features': 'log2',

            'random_state': 0

        }

    }

}



soft_models = {

    'RandomForestClassifier': {

        'model': RandomForestClassifier,

        'param': {

            'n_estimators': 167,

            'criterion': 'gini',

            'min_samples_leaf': 6,

            'max_features': 'log2',

            'random_state': 0

        }

    },

    'ExtraTreesClassifier': {

        'model': ExtraTreesClassifier,

        'param': {

            'n_estimators': 125,

            'criterion': 'gini',

            'min_samples_leaf': 5,

            'max_features': 'sqrt',

            'random_state': 0

        }

    },

    'AdaBoostClassifier': {

        'model': AdaBoostClassifier,

        'param': {

            'n_estimators': 198,

            'learning_rate': 0.9928786250309561,

            'random_state': 0

        }

    },

    'GradientBoostingClassifier': {

        'model': GradientBoostingClassifier,

        'param': {

            'n_estimators': 166,

            'learning_rate': 0.7479432334085875,

            'subsample': 0.9,

            'min_samples_leaf': 1,

            'max_features': 'log2',

            'random_state': 0

        }

    }

}



ground_truth = {

    "hard": {

        "train": y_hard_train,

        "test": y_hard_valid

    },

    "soft": {

        "train": y_soft_train,

        "test": y_soft_valid

    }

}



preds = {

    "hard": {

        "train": {},

        "test": {}

    },

    "soft": {

        "train": {},

        "test": {}

    }

}



# For ROC Curve

preds_proba = {

    "hard": {

        "train": {},

        "test": {}

    },

    "soft": {

        "train": {},

        "test": {}

    }

}
for current_model in hard_models:

    clf = hard_models[current_model]['model'](**hard_models[current_model]['param'])

    clf.fit(X_hard_train, y_hard_train)



    preds["hard"]["train"][current_model] = clf.predict(X_hard_train)

    preds["hard"]["test"][current_model] = clf.predict(X_hard_valid)



    preds_proba["hard"]["train"][current_model] = clf.predict_proba(X_hard_train)

    preds_proba["hard"]["test"][current_model] = clf.predict_proba(X_hard_valid)



for current_model in soft_models:

    clf = soft_models[current_model]['model'](**soft_models[current_model]['param'])

    clf.fit(X_soft_train, y_soft_train)



    preds["soft"]["train"][current_model] = clf.predict(X_soft_train)

    preds["soft"]["test"][current_model] = clf.predict(X_soft_valid)



    preds_proba["soft"]["train"][current_model] = clf.predict_proba(X_soft_train)

    preds_proba["soft"]["test"][current_model] = clf.predict_proba(X_soft_valid)
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
def metric_score(y_true, y_pred, y_pred_proba, title):

    accuracy = accuracy_score(y_true, y_pred)

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])

    roc_auc = auc(fpr, tpr)

    f1 = f1_score(y_true, y_pred)

    precision = precision_score(y_true, y_pred)

    recall = recall_score(y_true, y_pred)

    best_threshold = thresholds[np.argmax(tpr - fpr)]



    print("Metrics for " + title)

    print("Accuracy :", accuracy)

    print("ROC AUC  :", roc_auc)

    print("F1       :", f1)

    print("Precision:", precision)

    print("Recall   :", recall)

    print("Threshold:", best_threshold, "(Best)")

    print("\nFollowing output for copy-pasting into Excel:\n{},{},{},{},{}".format(accuracy, roc_auc, f1, precision, recall))
def plot_confusion_matrix(y_true, y_pred, y_pred_proba, title):

    plt.figure(figsize = (20,7))

    plt.subplots_adjust(right=1)

    plt.subplot(1, 2, 1)



    data = confusion_matrix(y_true, y_pred)

    df_cm = pd.DataFrame(data, columns=np.unique(y_true), index = np.unique(y_true))



    print('Confusion matrix for ' + title)

    print(df_cm)



    df_cm.index.name = 'Actual'

    df_cm.columns.name = 'Predicted'

    plt.title('Confusion matrix for ' + title)

    sns.set(font_scale=1.4) # Label size

    sns.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 16}) # Font size



    plt.subplot(1, 2, 2, facecolor='aliceblue')



    # Calculate ROC Curve, we take only the positive outcomes from the probabilities

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])

    roc_auc = auc(fpr, tpr)



    # We use Youden's J statistic for ease

    J = tpr - fpr

    idx = np.argmax(J)



    plt.title('ROC Curve : ' + title)

    plt.plot(fpr, tpr, 'r', label='AUC = %0.3f' % roc_auc)

    plt.scatter(fpr[idx], tpr[idx], marker='o', color='r', label='Best threshold = %0.3f' % thresholds[idx])

    plt.legend(loc='lower right')

    plt.plot([0, 1], [0, 1], 'b--')

    plt.xlim([-0.01, 1.01])

    plt.ylim([-0.01, 1.01])

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')



    plt.show()
for churn_type, churn_dict in preds.items():

    for feature_type, models in churn_dict.items():

        for model, results in models.items():

            y_valid = ground_truth[churn_type][feature_type]

            y_pred_proba = preds_proba[churn_type][feature_type][model]

            print('---')

            metric_score(y_valid, results, y_pred_proba, '{} churn ({}) [{}]'.format(churn_type, feature_type, model))

            print('')

            plot_confusion_matrix(y_valid, results, y_pred_proba, '{} churn ({}) [{}]'.format(churn_type, feature_type, model))

print('---')