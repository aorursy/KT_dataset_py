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

    show_feature_importance(clf, current_model, X_hard_train, y_hard_train, 'hard', 'all_features')



    # Hard with feature 5 removed

    clf.fit(X_hard_train_sans_5, y_hard_train)

    show_feature_importance(clf, current_model, X_hard_train_sans_5, y_hard_train, 'hard', 'sans_feature_5')

    

    # Soft with all features

    clf.fit(X_soft_train, y_soft_train)

    show_feature_importance(clf, current_model, X_soft_train, y_soft_train, 'soft', 'all_features')



    # Soft with feature 5 removed

    clf.fit(X_soft_train_sans_5, y_soft_train)

    show_feature_importance(clf, current_model, X_soft_train_sans_5, y_soft_train, 'soft', 'sans_feature_5')