import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')
import os
# Any results you write to the current directory are saved as output.
train_application_df = pd.read_csv('../input/application_train.csv')
test_application_df = pd.read_csv('../input/application_test.csv')
previous_application_df = pd.read_csv('../input/previous_application.csv')
all_application_df = pd.concat([train_application_df, test_application_df], axis=0)
categorical_features = []

for column in train_application_df.columns:
    if column == 'TARGET' or column == 'SK_ID_CURR':
        continue
    if train_application_df[column].dtype == 'object':
        all_application_df[column], _ = pd.factorize(all_application_df[column])
        categorical_features.append(column)
categorical_features = list(set(categorical_features))
print(len(categorical_features), categorical_features)
# Code for cross validation
def cross_validation(df):
    X = df[df['TARGET'].notna()]
    Y = X.pop('TARGET')
    if 'SK_ID_CURR' in df.columns:
        X.pop('SK_ID_CURR')
    num_fold = 5
    skf = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=2018)
    valid_scores = []
    train_scores = []
    
    for train_index, test_index in skf.split(X, Y):
        X_train, X_validation = X.iloc[train_index], X.iloc[test_index]
        y_train, y_validation = Y.iloc[train_index], Y.iloc[test_index]
        
        clf = LGBMClassifier(
            boosting_type='gbdt',
            objective='binary',
            n_estimators=1000,
            learning_rate=0.1,
            num_leaves=31,
            feature_fraction=0.8,
            subsample=0.8,
            max_depth=8,
            reg_alpha=1,
            reg_lambda=1,
            min_child_weight=40,
            random_state=2018,
            nthread=-1
            )
                   
        clf.fit(X_train, y_train, 
                eval_set=[(X_train, y_train), (X_validation, y_validation)], 
                eval_metric='auc',
                verbose = False,
                early_stopping_rounds=100
                )
        
        train_prediction = clf.predict_proba(X_train)[:, 1]
        train_score = roc_auc_score(y_train, train_prediction)
        train_scores.append(train_score)
        
        valid_prediction = clf.predict_proba(X_validation)[:, 1]
        valid_score = roc_auc_score(y_validation, valid_prediction)
        valid_scores.append(valid_score)
        
        print('Fold', train_score, valid_score, clf.best_iteration_)
    print('AUC mean:', np.mean(valid_scores), 'std:',np.std(valid_scores))
    
cross_validation(all_application_df)
all_features = list(all_application_df.columns)
all_features.remove('TARGET')
all_features.remove('SK_ID_CURR')

print(len(all_features))
print(all_features)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import matplotlib.pyplot as plt


# Code for cross validation
def cross_validation_with_reduction(df, reducer, scaler):
    X = df[df['TARGET'].notna()]
    Y = X.pop('TARGET')
    if 'SK_ID_CURR' in df.columns:
        X.pop('SK_ID_CURR')
    num_fold = 5
    
     # fill na
    for feature in all_features:
        X[feature] = X[feature].fillna(X[feature].mean())
    
    # scaling
    X[all_features] = pd.DataFrame(scaler.fit_transform(X[all_features]))
    
    skf = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=2018)
    valid_scores = []
    train_scores = []
    
    for train_index, test_index in skf.split(X, Y):
        X_train, X_validation = X.iloc[train_index], X.iloc[test_index]
        y_train, y_validation = Y.iloc[train_index], Y.iloc[test_index]
        
        
        reducer.fit(X_train[all_features], y_train)
        train_reduced_samples = pd.DataFrame(reducer.transform(X_train[all_features]))
        valid_reduced_samples = pd.DataFrame(reducer.transform(X_validation[all_features]))
        
        for feature in train_reduced_samples.columns:
            X_train[feature] = train_reduced_samples[feature].values
        
        for feature in valid_reduced_samples.columns:
            X_validation[feature] = valid_reduced_samples[feature].values
        
        clf = LGBMClassifier(
            boosting_type='gbdt',
            objective='binary',
            n_estimators=1000,
            learning_rate=0.1,
            num_leaves=31,
            feature_fraction=0.8,
            subsample=0.8,
            max_depth=8,
            reg_alpha=1,
            reg_lambda=1,
            min_child_weight=40,
            random_state=2018,
            nthread=-1
            )
                   
        clf.fit(X_train, y_train, 
                eval_set=[(X_train, y_train), (X_validation, y_validation)], 
                eval_metric='auc',
                verbose = False,
                early_stopping_rounds=100
                )
        
        train_prediction = clf.predict_proba(X_train)[:, 1]
        train_score = roc_auc_score(y_train, train_prediction)
        train_scores.append(train_score)
        
        valid_prediction = clf.predict_proba(X_validation)[:, 1]
        valid_score = roc_auc_score(y_validation, valid_prediction)
        valid_scores.append(valid_score)
        
        print('Fold', train_score, valid_score, clf.best_iteration_)
    print('AUC mean:', np.mean(valid_scores), 'std:',np.std(valid_scores))
from sklearn.decomposition import PCA
cross_validation_with_reduction(all_application_df, PCA(n_components=10), MinMaxScaler())
cross_validation_with_reduction(all_application_df, PCA(n_components=10), MinMaxScaler())
cross_validation_with_reduction(all_application_df, PCA(n_components=10), RobustScaler())
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
cross_validation_with_reduction(all_application_df, LDA(n_components=10), MinMaxScaler())
cross_validation_with_reduction(all_application_df, LDA(n_components=10), StandardScaler())
cross_validation_with_reduction(all_application_df, LDA(n_components=10), RobustScaler())
from sklearn.cross_decomposition import PLSSVD
cross_validation_with_reduction(all_application_df, PLSSVD(n_components=10), MinMaxScaler())
cross_validation_with_reduction(all_application_df, PLSSVD(n_components=10), StandardScaler())
cross_validation_with_reduction(all_application_df, PLSSVD(n_components=10), RobustScaler())
