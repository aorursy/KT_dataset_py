import sklearn



print('The scikit-learn version is {}.'.format(sklearn.__version__))
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
raw_hard_churn.corr()
plt.figure(figsize=(14, 7))

plt.title("Feature Correlation")

sns.heatmap(raw_hard_churn.loc[:, raw_hard_churn.columns != 'imei_name'].corr(), annot=True)
from sklearn.model_selection import train_test_split



def split_data(dataframe, result_column, test_size):

    X, y = dataframe.loc[:, (dataframe.columns != result_column) & (dataframe.columns != 'imei_name')], dataframe[result_column]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size)

    return X_train, X_valid, y_train, y_valid



X_hard_train, X_hard_valid, y_hard_train, y_hard_valid = split_data(raw_hard_churn, 'churn', 0.5)

X_soft_train, X_soft_valid, y_soft_train, y_soft_valid = split_data(raw_soft_churn, 'churn', 0.5)
X_hard_train.head()
y_hard_train.head()
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier



# Random state here is set for reproducibility

# Using default parameters



models = {

    'RandomForestClassifier': RandomForestClassifier,

    'ExtraTreesClassifier': ExtraTreesClassifier,

    'AdaBoostClassifier': AdaBoostClassifier,

    'GradientBoostingClassifier': GradientBoostingClassifier,

}



ground_truth = {

    "hard": {

        "all_features": {

            "train": y_hard_train,

            "test": y_hard_valid

        },

        "sans_feature_5": {

            "train": y_hard_train,

            "test": y_hard_valid

        }

    },

    "soft": {

        "all_features": {

            "train": y_soft_train,

            "test": y_soft_valid

        },

        "sans_feature_5": {

            "train": y_soft_train,

            "test": y_soft_valid

        }

    }

}



preds = {

    "hard": {

        "all_features": {

            "train": {},

            "test": {}

        },

        "sans_feature_5": {

            "train": {},

            "test": {}

        }

    },

    "soft": {

        "all_features": {

            "train": {},

            "test": {}

        },

        "sans_feature_5": {

            "train": {},

            "test": {}

        }

    }

}



X_hard_train_sans_5 = X_hard_train.loc[:, X_hard_train.columns != 'feature_5']

X_hard_valid_sans_5 = X_hard_valid.loc[:, X_hard_valid.columns != 'feature_5']



X_soft_train_sans_5 = X_soft_train.loc[:, X_soft_train.columns != 'feature_5']

X_soft_valid_sans_5 = X_soft_valid.loc[:, X_soft_valid.columns != 'feature_5']
from sklearn.inspection import permutation_importance



def show_feature_importance(model, model_name, X, y, churn_type, feature_type):

    feature_names = X.columns

    tree_feature_importances = model.feature_importances_

    sorted_idx = tree_feature_importances.argsort()

    

    y_ticks = np.arange(0, len(feature_names))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))

    ax1.barh(y_ticks, tree_feature_importances[sorted_idx])

    ax1.set_yticklabels(feature_names[sorted_idx])

    ax1.set_yticks(y_ticks)

    ax1.set_title("{} Feature Importances (MDI) [{}_{}]".format(model_name, churn_type, feature_type))

    

    result = permutation_importance(model, X, y, n_repeats=10, random_state=0, n_jobs=2)

    sorted_idx = result.importances_mean.argsort()



    ax2.boxplot(result.importances[sorted_idx].T, vert=False, labels=feature_names[sorted_idx])

    ax2.set_title("{} Permutation Importances (train set) [{}_{}]".format(model_name, churn_type, feature_type))

    plt.show()
for current_model in models:

    clf = models[current_model](random_state=0)

    

    # Hard with all features

    clf.fit(X_hard_train, y_hard_train)

    preds['hard']['all_features']['train'][current_model] = clf.predict(X_hard_train)

    preds['hard']['all_features']['test'][current_model] = clf.predict(X_hard_valid)

    # show_feature_importance(clf, current_model, X_hard_train, y_hard_train, 'hard', 'all_features')



    # Hard with feature 5 removed

    clf.fit(X_hard_train_sans_5, y_hard_train)

    preds['hard']['sans_feature_5']['train'][current_model] = clf.predict(X_hard_train_sans_5)

    preds['hard']['sans_feature_5']['test'][current_model] = clf.predict(X_hard_valid_sans_5)

    # show_feature_importance(clf, current_model, X_hard_train_sans_5, y_hard_train, 'hard', 'sans_feature_5')

    

    # Soft with all features

    clf.fit(X_soft_train, y_soft_train)

    preds['soft']['all_features']['train'][current_model] = clf.predict(X_soft_train)

    preds['soft']['all_features']['test'][current_model] = clf.predict(X_soft_valid)

    # show_feature_importance(clf, current_model, X_soft_train, y_soft_train, 'soft', 'all_features')



    # Soft with feature 5 removed

    clf.fit(X_soft_train_sans_5, y_soft_train)

    preds['soft']['sans_feature_5']['train'][current_model] = clf.predict(X_soft_train_sans_5)

    preds['soft']['sans_feature_5']['test'][current_model] = clf.predict(X_soft_valid_sans_5)

    # show_feature_importance(clf, current_model, X_soft_train_sans_5, y_soft_train, 'soft', 'sans_feature_5')
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
def metric_score(y_true, y_pred, title):

    accuracy = accuracy_score(y_true, y_pred)

    f1 = f1_score(y_true, y_pred)

    precision = precision_score(y_true, y_pred)

    recall = recall_score(y_true, y_pred)

    roc_auc = roc_auc_score(y_true, y_pred)

    

    print("Metrics for " + title)

    print("Accuracy  :", accuracy)

    print("ROC AUC   :", roc_auc)

    print("F1        :", f1)

    print("Precision :", precision)

    print("Recall    :", recall)
def plot_confusion_matrix(y_true, y_pred, title):

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



    fpr, tpr, _ = roc_curve(y_true, y_pred)

    roc_auc = auc(fpr, tpr)



    plt.title('ROC Curve : ' + title)

    plt.plot(fpr, tpr, 'r', label='AUC = %0.3f' % roc_auc)

    plt.legend(loc='lower right')

    plt.plot([0, 1], [0, 1], 'b--')

    plt.xlim([-0.01, 1.01])

    plt.ylim([-0.01, 1.01])

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')



    plt.show()
for churn_type, churn_dict in preds.items():

    for feature_type, data_types in churn_dict.items():

        for data_type, models in data_types.items():

            for model, results in models.items():

                y_valid = ground_truth[churn_type][feature_type][data_type]

                print('---')

                metric_score(y_valid, results, '{} churn ({}) [{}] [{}]'.format(churn_type, feature_type, model, data_type))

                print('')

                plot_confusion_matrix(y_valid, results, '{} churn ({}) [{}] [{}]'.format(churn_type, feature_type, model, data_type))

print('---')