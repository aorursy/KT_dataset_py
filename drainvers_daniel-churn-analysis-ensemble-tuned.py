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



hard_models = {

    'RandomForestClassifier': {

        'model': RandomForestClassifier,

        'param': {

            'n_estimators': 124,

            'criterion': 'entropy',

            'min_samples_leaf': 2,

            'max_features': 'log2',

            'random_state': 0

        }

    },

    'ExtraTreesClassifier': {

        'model': ExtraTreesClassifier,

        'param': {

            'n_estimators': 179,

            'criterion': 'entropy',

            'min_samples_leaf': 2,

            'max_features': 'log2',

            'random_state': 0

        }

    },

    'AdaBoostClassifier': {

        'model': AdaBoostClassifier,

        'param': {

            'n_estimators': 191,

            'learning_rate': 0.9824531051022739,

            'random_state': 0

        }

    },

    'GradientBoostingClassifier': {

        'model': GradientBoostingClassifier,

        'param': {

            'n_estimators': 198,

            'learning_rate': 0.6841850952084808,

            'subsample': 0.9,

            'min_samples_leaf': 3,

            'max_features': 'log2',

            'random_state': 0

        }

    }

}



soft_models = {

    'RandomForestClassifier': {

        'model': RandomForestClassifier,

        'param': {

            'n_estimators': 153,

            'criterion': 'entropy',

            'min_samples_leaf': 2,

            'max_features': 'log2',

            'random_state': 0

        }

    },

    'ExtraTreesClassifier': {

        'model': ExtraTreesClassifier,

        'param': {

            'n_estimators': 145,

            'criterion': 'entropy',

            'min_samples_leaf': 1,

            'max_features': 'log2',

            'random_state': 0

        }

    },

    'AdaBoostClassifier': {

        'model': AdaBoostClassifier,

        'param': {

            'n_estimators': 200,

            'learning_rate': 0.9774069535137836,

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

        "all_train": y_hard_train,

        "all_test": y_hard_valid,

        "wo5_train": y_hard_train,

        "wo5_test": y_hard_valid

    },

    "soft": {

        "all_train": y_soft_train,

        "all_test": y_soft_valid,

        "wo5_train": y_soft_train,

        "wo5_test": y_soft_valid

    }

}



preds = {

    "hard": {

        "all_train": {},

        "all_test": {},

        "wo5_train": {},

        "wo5_test": {}

    },

    "soft": {

        "all_train": {},

        "all_test": {},

        "wo5_train": {},

        "wo5_test": {}

    }

}



X_hard_train_sans_5 = X_hard_train.loc[:, X_hard_train.columns != 'feature_5']

X_hard_valid_sans_5 = X_hard_valid.loc[:, X_hard_valid.columns != 'feature_5']



X_soft_train_sans_5 = X_soft_train.loc[:, X_soft_train.columns != 'feature_5']

X_soft_valid_sans_5 = X_soft_valid.loc[:, X_soft_valid.columns != 'feature_5']
for current_model in hard_models:

    clf = hard_models[current_model]['model'](**hard_models[current_model]['param'])

    

    # Hard with all features

    print('Fitting {} [hard, all]...'.format(current_model))

    %timeit clf.fit(X_hard_train, y_hard_train)

    print('')



    print('Predicting {} [hard, all_train]...'.format(current_model))

    %timeit preds['hard']['all_train'][current_model] = clf.predict(X_hard_train)

    print('')



    print('Predicting {} [hard, all_test]...'.format(current_model))

    %timeit preds['hard']['all_test'][current_model] = clf.predict(X_hard_valid)

    print('---')



    # Hard with feature 5 removed

    print('Fitting {} [hard, wo5]...'.format(current_model))

    %timeit clf.fit(X_hard_train_sans_5, y_hard_train)

    print('')



    print('Predicting {} [hard, wo5_train]...'.format(current_model))

    %timeit preds['hard']['wo5_train'][current_model] = clf.predict(X_hard_train_sans_5)

    print('')



    print('Predicting {} [hard, wo5_test]...'.format(current_model))

    %timeit preds['hard']['wo5_test'][current_model] = clf.predict(X_hard_valid_sans_5)

    print('---')

    

for current_model in soft_models:

    clf = soft_models[current_model]['model'](**soft_models[current_model]['param'])

    

    # Soft with all features

    print('Fitting {} [soft, all]...'.format(current_model))

    %timeit clf.fit(X_soft_train, y_soft_train)

    print('')



    print('Predicting {} [soft, all_train]...'.format(current_model))

    %timeit preds['soft']['all_train'][current_model] = clf.predict(X_soft_train)

    print('')



    print('Predicting {} [soft, all_test]...'.format(current_model))

    %timeit preds['soft']['all_test'][current_model] = clf.predict(X_soft_valid)

    print('---')



    # Soft with feature 5 removed

    print('Fitting {} [soft, wo5]...'.format(current_model))

    %timeit clf.fit(X_soft_train_sans_5, y_soft_train)

    print('')

    

    print('Predicting {} [soft, wo5_train]...'.format(current_model))

    %timeit preds['soft']['wo5_train'][current_model] = clf.predict(X_soft_train_sans_5)

    print('')

    

    print('Predicting {} [soft, wo5_test]...'.format(current_model))

    %timeit preds['soft']['wo5_test'][current_model] = clf.predict(X_soft_valid_sans_5)

    print('---')
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
def metric_score(y_true, y_pred, title):

    accuracy = accuracy_score(y_true, y_pred)

    roc_auc = roc_auc_score(y_true, y_pred)

    f1 = f1_score(y_true, y_pred)

    precision = precision_score(y_true, y_pred)

    recall = recall_score(y_true, y_pred)

    

    print("Metrics for " + title)

    print("Accuracy :", accuracy)

    print("ROC AUC  :", roc_auc)

    print("F1       :", f1)

    print("Precision:", precision)

    print("Recall   :", recall)
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

    for feature_type, models in churn_dict.items():

        for model, results in models.items():

            y_valid = ground_truth[churn_type][feature_type]

            print('---')

            metric_score(y_valid, results, '{} churn ({}) [{}]'.format(churn_type, feature_type, model))

            print('')

            plot_confusion_matrix(y_valid, results, '{} churn ({}) [{}]'.format(churn_type, feature_type, model))

print('---')