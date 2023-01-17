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



import lightgbm as lgbm

import datetime

import sklearn.model_selection
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
pd_training.head()
pd_training_no_label = pd_training.drop('death_yn', axis='columns')

labels = pd_training['death_yn'].map({'Yes': 1, 'No': 0})

print('Records:', labels.size, '\nDeaths:', labels.sum())
pd_training_naive = pd_training_no_label.fillna({

    'cdc_report_dt': datetime.datetime(2020, 1, 1),

    'pos_spec_dt': datetime.datetime(2020, 1, 1),

    'onset_dt': datetime.datetime(2020, 1, 1),

    'sex': 'Unknown',

    'age_group': 'Unknown',

    'Race and ethnicity (combined)': 'Unknown',

})

pd_training_naive['cdc_report_dt'] = pd_training_naive['cdc_report_dt'].dt.strftime('%Y%m%d')

pd_training_naive['pos_spec_dt'] = pd_training_naive['pos_spec_dt'].dt.strftime('%Y%m%d')

pd_training_naive['onset_dt'] = pd_training_naive['onset_dt'].dt.strftime('%Y%m%d')

pd_training_naive = pd_training_naive.astype({

    'cdc_report_dt': 'int',

    'pos_spec_dt': 'int',

    'onset_dt': 'int',

})



map_yn_cols = {

    'Yes': 1,

    'No': 0,

    'Unknown': -1,

}

pd_training_naive['hosp_yn'] = pd_training_naive['hosp_yn'].map(map_yn_cols)

pd_training_naive['icu_yn'] = pd_training_naive['icu_yn'].map(map_yn_cols)

pd_training_naive['medcond_yn'] = pd_training_naive['medcond_yn'].map(map_yn_cols)



pd_training_naive['sex'] = pd_training_naive['sex'].map({'Male': 0, 'Female': 1, 'Other': 2, 'Unknown': 3})

pd_training_naive['current_status'] = pd_training_naive['current_status'].map({'Laboratory-confirmed case': 0, 'Probable Case': 1})

pd_training_naive['age_group'] = pd_training_naive['age_group'].map({

    '0 - 9 Years': 0,

    '10 - 19 Years': 1,

    '20 - 29 Years': 2,

    '30 - 39 Years': 3,

    '40 - 49 Years': 4,

    '50 - 59 Years': 5,

    '60 - 69 Years': 6,

    '70 - 79 Years': 7,

    '80+ Years': 8,

    'Unknown': -1,

})

pd_training_naive['Race and ethnicity (combined)'] = pd_training_naive['Race and ethnicity (combined)'].map({

    'American Indian/Alaska Native, Non-Hispanic': 0,

    'Asian, Non-Hispanic': 1,

    'Black, Non-Hispanic': 2,

    'Hispanic/Latino': 3,

    'Multiple/Other, Non-Hispanic': 4,

    'Native Hawaiian/Other Pacific Islander, Non-Hispanic': 5,

    'Unknown': 6,

    'White, Non-Hispanic': 7,

})

pd_training_naive.head()
training, holdout, training_labels, holdout_labels = sklearn.model_selection.train_test_split(

    pd_training_naive,

    labels,

    train_size=0.8,

    random_state=42,

)

print('Records in training:', len(training), '\nRecords in holdout:', len(holdout))
training_dset = lgbm.Dataset(

    training,

    training_labels,

)

valid_dset = lgbm.Dataset(

    holdout,

    holdout_labels,

)

lgbm_params = {

    'task': 'train',

    'objective': 'binary',

    'metric': 'binary_logloss',

    'learning_rate': 0.03,

    'num_leaves': 128,

    'seed': 42,

}

evals_naive = {}

history = lgbm.train(

    lgbm_params,

    training_dset,

    num_boost_round=800,

    verbose_eval=50,

    valid_sets=[valid_dset],

    valid_names=['valid_naive'],

    evals_result=evals_naive,

)
all_features = pd.concat(

    [

        pd_training_no_label.assign(train_test='Train'),

        pd_prediction.assign(train_test='Test'),

    ],

    axis='rows',

)

print('Records in stacked features:', len(all_features))
import sklearn.preprocessing



categorical_features = [

    'current_status',

    'sex',

    'age_group',

    'Race and ethnicity (combined)',

    'hosp_yn',

    'icu_yn',

    'medcond_yn',

]

all_features = all_features.fillna({

    cat_var: 'Missing'

    for cat_var in categorical_features

})

enc = sklearn.preprocessing.OneHotEncoder()

ohe_features = enc.fit_transform(all_features[categorical_features])
import re

ohe_vars = []

for cat_var, cat_var_categories in zip(categorical_features, enc.categories_):

    for category in cat_var_categories:

        colname = f'{cat_var}_{category}'

        colname_cleaner = lambda x:re.sub('[^A-Za-z0-9_]+', '_', x)

        ohe_vars.append(

            colname_cleaner(colname)

        )

print(ohe_vars)
all_features[ohe_vars] = ohe_features.toarray()

all_features.head()
date_features = [

    'cdc_report_dt',

    'pos_spec_dt',

    'onset_dt',

]

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





all_features.head()
# Add case counts

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

all_features.head()
# Add some simple PCA



import sklearn.decomposition

import sklearn.pipeline



features_for_pca = all_features.drop(

    categorical_features + date_features + ['id','train_test'],

    axis='columns',

)

pipeline = sklearn.pipeline.Pipeline(

    [

        ('normalizer', sklearn.preprocessing.StandardScaler()),

        ('pca', sklearn.decomposition.PCA(n_components=10)),

    ]

)

pca_features = pipeline.fit_transform(features_for_pca)

pca_cols = [f'pca_{num}' for num in range(1, 11)]

all_features[pca_cols] = pca_features

all_features.head()
pca_fit = pipeline.named_steps['pca']

print(pca_fit.explained_variance_ratio_)
# Drop our un-trainable features



all_features = all_features.drop(

    categorical_features + date_features + ['cdc_report_dt_week', 'pos_spec_dt_week', 'onset_dt_week'],

    axis='columns',

)
# Split back into train/test

train_engineered_features = all_features.loc[lambda df: df.train_test == 'Train'].drop('train_test', axis='columns')

test_engineered_features = all_features.loc[lambda df: df.train_test == 'Test'].drop('train_test', axis='columns')



train_engineered_features.head()
# Make sure we didn't break any row ordering in our feature engineering pipeline

assert (train_engineered_features.id == pd_training.id).all()



training_engineered, holdout_engineered = sklearn.model_selection.train_test_split(

    train_engineered_features,

    train_size=0.8,

    random_state=42,

)



training_engineered_dset = lgbm.Dataset(

    training_engineered,

    training_labels,

)

valid_engineered_dset = lgbm.Dataset(

    holdout_engineered,

    holdout_labels,

)

evals_engineered = {}

engineered_history = lgbm.train(

    lgbm_params,

    training_engineered_dset,

    num_boost_round=800,

    verbose_eval=50,

    valid_sets=[valid_engineered_dset],

    valid_names=['valid_engineered'],

    evals_result=evals_engineered,

)
evals = {**evals_naive, **evals_engineered}

lgbm.plot_metric(evals, figsize=(10, 8))
lgbm.plot_importance(history, figsize=(8, 6), title='Feature Importance - Naive Features')

lgbm.plot_importance(engineered_history, figsize=(8, 6), max_num_features=15, title='Feature Importance - Engineered Features')