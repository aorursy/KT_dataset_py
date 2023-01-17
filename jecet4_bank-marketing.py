import numpy as np

import pandas as pd

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OrdinalEncoder

from sklearn.impute import SimpleImputer

from lightgbm import LGBMClassifier

import lightgbm as lgbm

import datetime

import xgboost as xgb

from xgboost import plot_importance

import matplotlib.pyplot as plt

from numpy import sort



np.set_printoptions(formatter={"float_kind": lambda x: "%g" % x})



all_data = pd.read_csv('/kaggle/input/ml1920-bank-marketing/bank-classification.csv', index_col=0)



# Have a peek on how the dataset looks like

print(all_data.head())
# Check which features are object

print(all_data.select_dtypes(include='object').columns)



obj_cols = list(set(all_data.select_dtypes(include='object').columns) - set(['y']))

nominal_cols = list(set(all_data.columns) - set(obj_cols) - set(['y']))
# Impute missing data (on both obj_cols and nominal_cols)

imputer = SimpleImputer(missing_values='unknown', strategy='most_frequent')

all_data[obj_cols] = imputer.fit_transform(all_data[obj_cols])



imputer = SimpleImputer(strategy='mean')

all_data[nominal_cols] = imputer.fit_transform(all_data[nominal_cols])

# Feature engineering

all_data.contact_date = all_data.contact_date.map(lambda x: pd.to_datetime(x))

all_data.birth_date = all_data.birth_date.map(lambda x: pd.to_datetime(x))

all_data['contact_year'] = all_data.contact_date.map(lambda x: x.year)

all_data['contact_month'] = all_data.contact_date.map(lambda x: x.month)

all_data['contact_day'] = all_data.contact_date.map(lambda x: x.day)

all_data['birth_year'] = all_data.birth_date.map(lambda x: x.year)

all_data['age_in_days'] = all_data.birth_date.map(lambda x: (pd.to_datetime(datetime.datetime.today().strftime('%Y%m%d')) - x).days)

all_data['days_since_contact'] = all_data.contact_date.map(lambda x: (pd.to_datetime(datetime.datetime.today().strftime('%Y%m%d')) - x).days)

all_data['no_contact'] = all_data.pdays.map(lambda x: int(x == 999))

all_data['age_*_days_since_contact'] = all_data[['days_since_contact', 'age_in_days']].apply(lambda row: row.days_since_contact * row.age_in_days, axis=1)
# Why do we use id as a feature?

# According to feature importance (and empirical proofs), it has a great influence on score

# Probably, because it's correlated with contact_date (the biger, the id)

# But it's way more important than contact_date, so there's much be more into it

all_data['id_'] = pd.Series(all_data.index)



# Drop no longer needed columns

all_data.drop(['birth_date', 'contact_date'], inplace=True, axis=1)



obj_cols.remove('birth_date')

obj_cols.remove('contact_date')
# Separate object cols to ones for ordinal and one-hot encoding

ord_cols = list(all_data[['education', 'job', 'marital', 'loan', 'housing', 'poutcome']].columns)

oh_cols = list(set(obj_cols) - set(ord_cols))



# Create Ordnungs on ordinal cats

# Ordnungs come off the top of my head - I just assumed this would be logical (no further research required, as they hardly affect the outcome at all)

# To check values for a feature, it's sufficient to call all_data['feature'].unique(), but there're a lot of them here, so I won't do that now

education_cats = ['illiterate', 'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'professional.course', 'university.degree']

job_cats = ['unemployed', 'housemaid', 'student', 'retired', 'blue-collar', 'admin.', 'entrepreneur', 'technician', 'services', 'self-employed', 'management']

housing_cats = ['no', 'yes']

loan_cats = ['yes', 'no']

marital_cats = ['divorced', 'single', 'married']

poutcome_cats = ['failure', 'nonexistent', 'success']



cats = [education_cats, job_cats, marital_cats, loan_cats, housing_cats, poutcome_cats]
# Deal with ordinal features

col_labeler = ColumnTransformer([('ordinal_encoder', OrdinalEncoder(categories=cats), ord_cols)])

ord_encoded_cols = pd.DataFrame(col_labeler.fit_transform(all_data), index=all_data.index, columns=['edu', 'job1', 'mari', 'loan1', 'housing1', 'poutcome1'])



# Deal with one-hot encoding

oh_encoder = OneHotEncoder(sparse=False)

col_encoder = ColumnTransformer([('oh_encoder', oh_encoder, oh_cols)])

oh_encoded_cols = pd.DataFrame(col_encoder.fit_transform(all_data), index=all_data.index)
# Add polynomials

# (For most important features, and some other nominal)

all_data['campaign_squared'] = all_data.campaign.map(lambda x: x ** 2)

all_data['days_since_contact_squared'] = all_data.days_since_contact.map(lambda x: x ** 2)

all_data['days_since_contact_cubic'] = all_data.days_since_contact.map(lambda x: x ** 3)

all_data['age_in_days_squared'] = all_data.age_in_days.map(lambda x: x ** 2)

all_data['previous_squared'] = all_data.previous.map(lambda x: x ** 2)

all_data['id_squared'] = all_data['id_'].map(lambda x: x ** 2)
# Drop obj cols

all_data.drop(obj_cols, axis=1, inplace=True)



# Concat with encoded obj cols and polynomial cols

all_data = pd.concat([all_data, oh_encoded_cols, ord_encoded_cols], axis=1)



# Drop the least improtant features (according to xgb's feature importance plot)

all_data.drop(['birth_year', 'housing1', 'contact_year', 'loan1', 'previous_squared', 'no_contact'], inplace=True, axis=1)
# Split data to train, y and submit

submit_data = all_data[all_data.y == 'unknown']

submit_data = submit_data.drop(labels='y', axis=1)

y = all_data[all_data.y != 'unknown'].y

y = y.apply(lambda y: int(y == 'yes'))

train_data = all_data[all_data.y != 'unknown'].drop(labels='y', axis=1)

# How to choose best params?

# Well, I read the docs (Python API) and made a bunch of experiments, checking which combination would perform best on cv

# Params for xgb

booster_param = {'booster': 'dart',

         'max_depth': 6,

         'learning_rate': 0.05,

         'objective': 'rank:pairwise',

         'sample_type': 'uniform',

         'normalize_type': 'forest',

         'rate_drop': 0.1,

         'skip_drop': 0.5,

         'n_estimators': 400,

         'eval_metric': 'auc',

         'num_boost_round': 200,

         'lambda': 1.0,

         'min_child_weight': 20,

         'subsample': 0.6,

         'gamma': 0.5,

         'colsample_bytree': 0.6

        }
# Train xgb

cv_scores = xgb.cv(booster_param, xgb.DMatrix(train_data, label=y), nfold=5, metrics='auc')

print("CV scores for xgb:")

print(cv_scores)

print('\n')

trained_model = xgb.train(booster_param, xgb.DMatrix(train_data, label=y))



xgb.plot_importance(trained_model)

plt.show()
# Params for lgbm

params = {

    'objective': 'cross_entropy_lambda',

    'boosting': 'dart',

    'num_iterations': 300,

    'learning_rate': 0.05,

    'num_leaves': 32,

    #'max_depth': 5,

    'bagging_fraction': 0.7,

    'bagging_freq': 2,

    'feature_fraction': 0.8,

    'lambda': 2.0,

    'max_bin': 256,

    'metric': 'auc'

}
# Train lgbm

trained_model = lgbm.train(params, lgbm.Dataset(train_data, label=y))

scores = lgbm.cv(params, lgbm.Dataset(train_data, label=y), nfold=5)

print('Mean cv score for lgbm:')

print(np.mean(scores['auc-mean']))

print('\n')



lgbm.plot_importance(trained_model)

plt.show()
# Generate predictions

# output = pd.DataFrame(data=trained_model.predict(xgb.DMatrix(submit_data)), index=submit_data.index, columns=['y'])

output = pd.DataFrame(data=trained_model.predict(submit_data), index=submit_data.index, columns=['y'])

output.to_csv('submission.csv')

print(output)