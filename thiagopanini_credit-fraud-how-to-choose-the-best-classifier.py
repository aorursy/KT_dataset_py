# Stsandard libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from datetime import datetime
import time

# Utilities
from viz_utils import *
from ml_utils import *
from custom_transformers import *

# Pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Modeling
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, cross_val_predict, \
                                    learning_curve
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, \
    accuracy_score, precision_score, recall_score, f1_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
# Data path
df_ori = import_data('../input/creditcardfraud/creditcard.csv', optimized=True)
df_ori.columns = [col.lower() for col in df_ori.columns]

# Results
print(f'Data dimension: {df_ori.shape}')
df_ori.head()
# Target class balance
fig, ax = plt.subplots(figsize=(7, 7))
label_names = ['Non-Fraud', 'Fraud']
color_list = ['darkslateblue', 'crimson']
text = f'Total\n{len(df_ori)}'
title = 'Target Class Balance'

# Visualizing it through a donut chart
donut_plot(df_ori, col='class', ax=ax, label_names=label_names, colors=color_list, title=title, text=text)
df_overview = data_overview(df_ori, corr=True, label_name='class')
df_overview.head(20)
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(13, 6))
target_correlation_matrix(data=df_ori, label_name='class', corr='positive', ax=axs[0])
target_correlation_matrix(data=df_ori, label_name='class', corr='negative', ax=axs[1])

plt.tight_layout()
plt.show()
numplot_analysis(df_ori.drop('class', axis=1), fig_cols=3)
# Features to be used on dataprep pipeline
features_ori = list(df_ori.drop('time', axis=1).columns)

# Construction a pre-processing pipeline from columns_transformers.py
pre_processing_pipe = Pipeline([
    ('selector', FeatureSelection(features=features_ori)),
    ('dup_dropper', DropDuplicates()),
    ('splitter', SplitData(target='class'))
])

# Executing the pipeline
X_train, X_test, y_train, y_test = pre_processing_pipe.fit_transform(df_ori)
model_features = list(X_train.columns)

# Looking at the results
print(f'Dimensões de X_train: {X_train.shape}')
print(f'Dimensões de y_train: {y_train.shape}')
print(f'\nDimensões de X_test: {X_test.shape}')
print(f'Dimensões de y_test: {y_test.shape}')
# Splitting the data by the dtype
num_attribs, cat_attribs = split_cat_num_data(X_train)
print(f'Total of numerical features: {len(num_attribs)}')
print(f'Total of categorical features: {len(cat_attribs)}')
# Preparing a dictionary to feed the ClassifiersAnalysis class
set_prep = {
    'X_train_prep': X_train.values,
    'X_test_prep': X_test.values,
    'y_train': y_train,
    'y_test': y_test
}
# Logistic Regression hyperparameters
logreg_param_grid = {
    'C': np.linspace(0.1, 10, 20),
    'penalty': ['l1', 'l2'],
    'class_weight': ['balanced', None],
    'random_state': [42],
    'solver': ['liblinear']
}

# Decision Trees hyperparameters
tree_param_grid = {
    'criterion': ['entropy', 'gini'],
    'max_depth': [3, 5, 10, 20],
    'max_features': np.arange(1, X_train.shape[1]),
    'class_weight': ['balanced', None],
    'random_state': [42]
}

# Random Forest hyperparameters
forest_param_grid = {
    'bootstrap': [True, False],
    'max_depth': [3, 5, 10, 20, 50],
    'n_estimators': [50, 100, 200, 500],
    'random_state': [42],
    'max_features': ['auto', 'sqrt'],
    'class_weight': ['balanced', None]
}

# LightGBM hyperparameters
lgbm_param_grid = {
    'num_leaves': list(range(8, 92, 4)),
    'min_data_in_leaf': [10, 20, 40, 60, 100],
    'max_depth': [3, 4, 5, 6, 8, 12, 16],
    'learning_rate': [0.1, 0.05, 0.01, 0.005],
    'bagging_freq': [3, 4, 5, 6, 7],
    'bagging_fraction': np.linspace(0.6, 0.95, 10),
    'reg_alpha': np.linspace(0.1, 0.95, 10),
    'reg_lambda': np.linspace(0.1, 0.95, 10),
}

lgbm_fixed_params = {
    'application': 'binary',
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': 'true',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'learning_rate': 0.05,
    'verbose': 0
}
# Preparando set de classificadores
set_classifiers = {
    'LogisticRegression': {
        'model': LogisticRegression(),
        'params': logreg_param_grid
    },
    'DecisionTrees': {
        'model': DecisionTreeClassifier(),
        'params': tree_param_grid
    },
    'RandomForest': {
        'model': RandomForestClassifier(),
        'params': forest_param_grid
    },
    'LightGBM': {
        'model': lgb.LGBMClassifier(**lgbm_fixed_params),
        'params': lgbm_param_grid
    }
}
# Instanciando classe e treinando set de classificadores
clf_tool = BinaryClassifiersAnalysis()
clf_tool.fit(set_classifiers, X_train, y_train, random_search=True, cv=3, verbose=5)
df_performances = clf_tool.evaluate_performance(X_train, y_train, X_test, y_test, cv=3)
cm = sns.light_palette('cornflowerblue', as_cmap=True)
df_performances.reset_index(drop='True').style.background_gradient(cmap=cm)
fig, ax = plt.subplots(figsize=(13, 12))
lgbm_feature_importance = clf_tool.feature_importance_analysis(model_features, specific_model='LightGBM', ax=ax)
plt.show()
clf_tool.plot_roc_curve()
clf_tool.plot_score_distribution('LightGBM')
# Separação por faixa
clf_tool.plot_score_bins('LightGBM', bin_range=0.1)
# Applying undersampling
rus = RandomUnderSampler()
X_train_under, y_train_under = rus.fit_sample(X_train, y_train)

# Training new classifiers using undersampling
undersamp_approach= '_undersamp'
clf_tool.fit(set_classifiers, X_train_under, y_train_under, random_search=True, cv=3, approach=undersamp_approach)
df_performances = clf_tool.evaluate_performance(X_train_under, y_train_under, X_test, y_test, cv=3)
cm = sns.light_palette('cornflowerblue', as_cmap=True)
df_performances.reset_index(drop='True').style.background_gradient(cmap=cm)
clf_tool.plot_roc_curve()
clf_tool.plot_score_distribution('LightGBM_undersamp')
clf_tool.plot_score_bins('LightGBM_undersamp', bin_range=0.1)
lgbm_set_classifier = {}
lgbm_set_classifier['LightGBM'] = set_classifiers['LightGBM']
# Applying oversampling
sm = SMOTE(sampling_strategy='minority', random_state=42)
X_train_over, y_train_over = sm.fit_sample(X_train, y_train)

# Treinando novo modelo após undersampling
oversamp_approach= '_oversamp'
clf_tool.fit(lgbm_set_classifier, X_train_over, y_train_over, random_search=True, cv=3, approach=oversamp_approach)
df_performances = clf_tool.evaluate_performance(X_train_over, y_train_over, X_test, y_test, cv=3)
cm = sns.light_palette('cornflowerblue', as_cmap=True)
df_performances.reset_index(drop='True').style.background_gradient(cmap=cm)
clf_tool.plot_roc_curve()
clf_tool.plot_score_distribution('LightGBM_oversamp')
clf_tool.plot_score_bins('LightGBM_oversamp', bin_range=0.1)
df_performances.query('approach == "Teste"').sort_values(by='auc', ascending=False)
