# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import typing

import re

import sklearn.decomposition

import sklearn.pipeline

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

from lightgbm import LGBMClassifier

from lightgbm import plot_importance

from sklearn.metrics import classification_report

import datetime
# Functionalize feature engineering. 

# why: there are some things that are extremely dependent on the available data, feature engineering like

# PCA or other dimensionality reduction techniques should be included inside the cross validation loop

# This is just functionalized from @ben-copeland's feature engineering



def engineer_features(pd_training_no_label, pd_prediction):



    categorical_features = [

        'current_status',

        'sex',

        'age_group',

        'Race and ethnicity (combined)',

        'hosp_yn',

        'icu_yn',

        'medcond_yn',

    ]

    date_features = [

        'cdc_report_dt',

        'pos_spec_dt',

        'onset_dt',

    ]

    

    all_features = pd.concat(

        [

            pd_training_no_label.assign(train_test='Train'),

            pd_prediction.assign(train_test='Test'),

        ],

        axis='rows',

    )

    n_components = 10

        

    all_features = all_features.fillna({

        cat_var: 'Missing'

        for cat_var in categorical_features

    })

    enc = sklearn.preprocessing.OneHotEncoder()

    ohe_features = enc.fit_transform(all_features[categorical_features])

    

    ohe_vars = []

    for cat_var, cat_var_categories in zip(categorical_features, enc.categories_):

        for category in cat_var_categories:

            colname = f'{cat_var}_{category}'

            colname_cleaner = lambda x:re.sub('[^A-Za-z0-9_]+', '_', x)

            ohe_vars.append(

                colname_cleaner(colname)

            )

    all_features[ohe_vars] = ohe_features.toarray()

    

 

    



    all_features = all_features.fillna({

        date_feature: datetime.datetime(2020, 1, 1)

        for date_feature in date_features

    })

    date_first_us_case = datetime.datetime(2020, 1, 21)

    for date_feature in date_features:

        all_features[f'{date_feature}_day'] = all_features[date_feature].dt.day

        all_features[f'{date_feature}_month'] = all_features[date_feature].dt.month

        all_features[f'{date_feature}_week'] = all_features[date_feature].dt.isocalendar().week

        all_features[f'{date_feature}_dayofweek'] = all_features[date_feature].dt.dayofweek

        all_features[f'{date_feature}_dayofyear'] = all_features[date_feature].dt.dayofyear

        all_features[f'{date_feature}_days_since_first_case'] = (all_features[date_feature] - date_first_us_case).dt.days

    

    # Novel date features



    all_features['days_diff_cdc_and_pos_spec'] = (all_features['cdc_report_dt'] - all_features['pos_spec_dt']).dt.days

    all_features['days_diff_cdc_and_onset'] = (all_features['cdc_report_dt'] - all_features['onset_dt']).dt.days

    all_features['days_diff_pos_spec_and_onset'] = (all_features['pos_spec_dt'] - all_features['onset_dt']).dt.days

    

    num_cases_by_date = all_features.groupby('cdc_report_dt_days_since_first_case').size()

    cumul_cases = num_cases_by_date.cumsum()

    cumul_cases.name = 'cumulative_cases'



    num_cases_by_week = all_features.groupby('cdc_report_dt_week').size()

    num_cases_by_week.name = 'cases_this_week'



    all_features = all_features.join(

        cumul_cases,

        on='cdc_report_dt_days_since_first_case',

    ).join(

        num_cases_by_week,

        on='cdc_report_dt_week',

    )    

    

    features_for_pca = all_features.drop(

        categorical_features + date_features + ['id', 'train_test'],

        axis='columns',

    )

    pipeline = sklearn.pipeline.Pipeline(

        [

            ('normalizer', sklearn.preprocessing.StandardScaler()),

            ('pca', sklearn.decomposition.PCA(n_components=n_components)),

        ]

    )

    pca_features = pipeline.fit_transform(features_for_pca)

    pca_cols = [f'pca_{num}' for num in range(1, n_components+1)]

    all_features[pca_cols] = pca_features

    

    pca_fit = pipeline.named_steps['pca']

    

    all_features = all_features.drop(

        categorical_features + date_features + ['cdc_report_dt_week', 'pos_spec_dt_week', 'onset_dt_week'],

        axis='columns',

    )

    pd_train = all_features.loc[lambda df: df.train_test == 'Train'].drop('train_test', axis='columns')

    pd_test = all_features.loc[lambda df: df.train_test == 'Test'].drop('train_test', axis='columns')



    return pd_train, pd_test



# Load data and split off label

date_cols = [

    'cdc_report_dt',

    'pos_spec_dt',

    'onset_dt',

]



pd_training = pd.read_csv(

    r'/kaggle/input/health-hackathon-prep/training.csv',

    parse_dates=date_cols,

)

pd_prediction = pd.read_csv(

    r'/kaggle/input/health-hackathon-prep/testing.csv',

    parse_dates=date_cols,

)

pd_training_no_label = pd_training.drop('death_yn', axis='columns')

labels = pd_training['death_yn'].map({'Yes': 1, 'No': 0})

pd_training_engineered, pd_test_engineered = engineer_features(pd_training_no_label, pd_prediction)

param_grid = {

    'num_leaves': [

        64,

        128,

        256

    ],

    'max_depth': [

        -1, 

        10, 

        20

    ],

    'n_estimators': [

        150, 

        200,

        400

    ],

    'learning_rate': [

        0.01,

        0.03,

        0.05,

        0.1, 

    ],

    'random_state': [42],

}

scv = StratifiedKFold(n_splits = 5, random_state=42, shuffle=True)



clf = GridSearchCV(estimator=LGBMClassifier(), param_grid=param_grid, cv=scv, n_jobs=-1, verbose=50, scoring='neg_log_loss')

clf.fit(pd_training_engineered, labels)

best_clf = clf.best_estimator_.fit(pd_training_engineered, labels)
plot_importance(clf.best_estimator_, max_num_features=15)
clf.cv_results_
clf.predict(pd_test_engineered)