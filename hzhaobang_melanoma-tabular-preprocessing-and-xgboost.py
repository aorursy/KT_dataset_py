import pandas as pd
import numpy as np

import os
import random
import re
import math
import time

# modules forxgboost modeling
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, cross_validate
from sklearn.metrics import roc_auc_score, roc_curve

# ignore warnings for clearner ouputs
import warnings
warnings.filterwarnings('ignore')
seed = 42
random.seed(42)
np.random.seed(42)
# setting path, change path here for other melanoma dataset

base_path = '/kaggle/input/siim-isic-melanoma-classification'
train_img_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'
test_img_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/test/'
img_stats_path = '/kaggle/input/melanoma2020imgtabular'
train = pd.read_csv(os.path.join(base_path, 'train.csv'))
test = pd.read_csv(os.path.join(base_path, 'test.csv'))
sample = pd.read_csv(os.path.join(base_path, 'sample_submission.csv'))
f'train features: {train.columns.tolist()}', f'test_featuers: {test.columns.tolist()};'
# do some simple renaming for complecatec feature name

train.columns = [
    'img_name', 'id', 'sex', 'age', 'location', 'diagnosis',
    'benign_malignant', 'target'
]
test.columns = ['img_name', 'id', 'sex', 'age', 'location']
train.head()
test.tail()
# now I'm using my own code for simplicity
def fill_missing(df):
    '''The strategy here is:
    1. For cat features we fill 'unknown' if catgories are many else mode
    2. For con features we fill the median
    '''
    df = df.copy()
    # fill nan values
    df.location.fillna(value='unknown', inplace=True)
    df.sex.fillna(value=train.sex.mode()[0], inplace=True)
    df.age.fillna(value=df.age.median(), inplace=True)

    return df
train = fill_missing(train)
test = fill_missing(test)
train.isnull().any(), test.isnull().any()
# Loading lanscape data

train40 = pd.read_csv('../input/melanoma2020imgtabular/train40Features.csv')
test40 = pd.read_csv('../input/melanoma2020imgtabular/test40Features.csv')

trainmet = pd.read_csv('../input/melanoma2020imgtabular/trainMetrics.csv')
testmet = pd.read_csv('../input/melanoma2020imgtabular/testMetrics.csv')
# drop duplicate data from landscape dataset
train40.drop(['sex', 'age_approx', 'anatom_site_general_challenge'],
             axis=1,
             inplace=True)
test40.drop(['sex', 'age_approx', 'anatom_site_general_challenge'],
            axis=1,
            inplace=True)

# merging both datasets
train = pd.concat([train, train40, trainmet], axis=1)
test = pd.concat([test, test40, testmet], axis=1)
train.head()
# def label_encoding(df):
#     df = df.copy()
#     # encode labels
#     location = LabelEncoder()
#     sex = LabelEncoder()
#     location_data = location.fit_transform(df.location)
#     sex_data = sex.fit_transform(df.sex)
#     df.location = location_data
#     df.sex = sex_data
    
#     return df

# train = label_encoding(train)
# test = label_encoding(test)
def dummy_encoding(df):
    df = df.copy()
    
    # dummy encoding for label sex and location
    sex_dummies = pd.get_dummies(df.sex, prefix='sex')
    location_dummies = pd.get_dummies(df.location, prefix='location')
    df = pd.concat([df, sex_dummies], axis=1)
    df = pd.concat([df, location_dummies], axis=1)
    
    return df
train = dummy_encoding(train)
test = dummy_encoding(test)
train.head()
train.drop(['sex','img_name','id','diagnosis','benign_malignant', 'location'], axis=1, inplace=True)
test.drop(['sex','img_name','id', 'location'], axis=1, inplace=True)
train.head()
# input and output for modelling

X = train.drop('target', axis=1)
y = train.target
# 5 stratified KFold with holdout validating

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state=seed)
cv = StratifiedKFold(5, shuffle=True, random_state=seed)
X_train.head()
# drop features for overfitting
# X_train.drop(['n_images', 'image_size','width','height','total_pixels','reds','blues','greens','mean_colors', 'age_min', 'age_max'], axis=1, inplace=True)
# test.drop(['n_images', 'image_size','width','height','total_pixels','reds','blues','greens','mean_colors', 'age_min', 'age_max'], axis=1, inplace=True)
xg = xgb.XGBClassifier(
    n_estimators=750,
    min_child_weight=0.81,
    learning_rate=0.025,
    max_depth=2,
    subsample=0.80,
    colsample_bytree=0.42,
    gamma=0.10,
    random_state=42,
    n_jobs=-1,
)
estimators = [xg]
def model_check(X_train, y_train, estimators, cv):
    model_table = pd.DataFrame()
    row_index = 0
    
    for est in estimators:
        MLA_name = est.__class__.__name__
        model_table.loc[row_index, 'Model Name'] = MLA_name
        
        cv_results = cross_validate(est,
                                    X_train,
                                    y_train,
                                    cv=cv,
                                    scoring='roc_auc',
                                    return_train_score=True,
                                    n_jobs=-1)
        
        model_table.loc[row_index,
                        'Train roc Mean'] = cv_results['train_score'].mean()
        model_table.loc[row_index,
                        'Test roc Mean'] = cv_results['test_score'].mean()
        model_table.loc[row_index, 'Test Std'] = cv_results['test_score'].std()
        model_table.loc[row_index, 'Time'] = cv_results['fit_time'].mean()
        
        row_index += 1
    
    model_table.sort_values(by=['Test roc Mean'],
                           ascending=False,
                           inplace=True)
    
    return model_table
raw_models = model_check(X_train, y_train, [xg], cv)
display(raw_models)
# xgboost predict

xg.fit(X_train, y_train)
predictions = xg.predict_proba(test)[:, 1]
# create submission file with two meta features required

meta_df = pd.DataFrame(columns=['image_name', 'target'])
meta_df['image_name'] = sample['image_name']
meta_df['target'] = predictions
meta_df.to_csv('xgboost_meta_simplified.csv', header=True, index=False)