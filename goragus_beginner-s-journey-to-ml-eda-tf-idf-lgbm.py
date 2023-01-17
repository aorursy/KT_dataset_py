# import libraries
import os

import math
import datetime

import tensorflow as tf
from tensorflow.python.data import Dataset

from scipy import stats
from scipy.sparse import hstack, csr_matrix
from mlxtend.preprocessing import minmax_scaling
import seaborn as sns

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pandas_profiling as pp

from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
from textblob import TextBlob

import xgboost as xgb
import lightgbm as lgb

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
# read CSV files
train = pd.read_csv("../input/train.csv", sep=",", parse_dates=['project_submitted_datetime'])
resources = pd.read_csv("../input/resources.csv", sep=",")
test = pd.read_csv("../input/test.csv", sep=",", parse_dates=['project_submitted_datetime'])
# extract features from resource data and merge with train and test
resources['cost'] = resources['quantity'] * resources['price']
resources_aggregated = resources.groupby('id').agg({'description': ['nunique'], 'quantity': ['sum'], 'cost': ['mean', 'sum']})
resources_aggregated.columns = ['unique_items', 'total_quantity', 'mean_cost', 'total_cost']
resources_aggregated.reset_index(inplace=True)

train = pd.merge(train, resources_aggregated, how='left', on='id')
test = pd.merge(test, resources_aggregated, how='left', on='id')
# combine essay1+2, essay 3+4 for proposals before May 7, 2017
train.loc[train.project_submitted_datetime.dt.date < datetime.date(2016, 5, 7), 'project_essay_1'] = train.loc[train.project_submitted_datetime.dt.date < datetime.date(2016, 5, 7), 'project_essay_1'] + ' ' + train.loc[train.project_submitted_datetime.dt.date < datetime.date(2016, 5, 7), 'project_essay_2']
train.loc[train.project_submitted_datetime.dt.date < datetime.date(2016, 5, 7), 'project_essay_2'] = train.loc[train.project_submitted_datetime.dt.date < datetime.date(2016, 5, 7), 'project_essay_3'] + ' ' + train.loc[train.project_submitted_datetime.dt.date < datetime.date(2016, 5, 7), 'project_essay_4']
train.drop(['project_essay_3', 'project_essay_4'], axis=1, inplace=True)

test.loc[test.project_submitted_datetime.dt.date < datetime.date(2016, 5, 7), 'project_essay_1'] = test.loc[test.project_submitted_datetime.dt.date < datetime.date(2016, 5, 7), 'project_essay_1'] + ' ' + test.loc[test.project_submitted_datetime.dt.date < datetime.date(2016, 5, 7), 'project_essay_2']
test.loc[test.project_submitted_datetime.dt.date < datetime.date(2016, 5, 7), 'project_essay_2'] = test.loc[test.project_submitted_datetime.dt.date < datetime.date(2016, 5, 7), 'project_essay_3'] + ' ' + test.loc[test.project_submitted_datetime.dt.date < datetime.date(2016, 5, 7), 'project_essay_4']
test.drop(['project_essay_3', 'project_essay_4'], axis=1, inplace=True)

# replacing symbols which appeared due to formatting
train['project_essay_1'] = train['project_essay_1'].apply(lambda x: x.replace('\\r', ' ').replace('\\n', ' ').replace('  ', ' '))
train['project_essay_2'] = train['project_essay_2'].apply(lambda x: x.replace('\\r', ' ').replace('\\n', ' ').replace('  ', ' '))

test['project_essay_1'] = test['project_essay_1'].apply(lambda x: x.replace('\\r', ' ').replace('\\n', ' ').replace('  ', ' '))
test['project_essay_2'] = test['project_essay_2'].apply(lambda x: x.replace('\\r', ' ').replace('\\n', ' ').replace('  ', ' '))

# replacing symbols in title and summary
train['project_resource_summary'] = train['project_resource_summary'].apply(lambda x: x.replace('\\r', ' ').replace('\\n', ' ').replace('  ', ' '))
train['project_title'] = train['project_title'].apply(lambda x: x.replace('\\r', ' ').replace('\\n', ' ').replace('  ', ' '))

test['project_resource_summary'] = test['project_resource_summary'].apply(lambda x: x.replace('\\r', ' ').replace('\\n', ' ').replace('  ', ' '))
test['project_title'] = test['project_title'].apply(lambda x: x.replace('\\r', ' ').replace('\\n', ' ').replace('  ', ' '))
# define bins and graph binned experience
bins = [0., 2., 10.,]
train["experience"] = np.digitize(train['teacher_number_of_previously_posted_projects'], bins=bins)

fig, ax1 = plt.subplots(figsize=(8, 4))
plt.title("Experience vs Approval")
sns.pointplot(x="experience", y="project_is_approved", data=train, ci=95, ax=ax1)
ax1.set_ylabel('Approval rate')
# define quantile function
def get_quantile_based_boundaries(feature_values, num_buckets):
  boundaries = np.arange(1.0, num_buckets) / num_buckets
  quantiles = feature_values.quantile(boundaries)
  return [quantiles[q] for q in quantiles.keys()]
# transform unique_items to quantile number and graph
quantiles = get_quantile_based_boundaries(train['unique_items'], 6)
train["unique_items_binned"] = np.digitize(train['unique_items'], bins=quantiles)
fig, ax1 = plt.subplots(figsize=(8, 4))
plt.title("unique_items vs Approval")
sns.pointplot(x="unique_items_binned", y="project_is_approved", data=train, ci=95, ax=ax1)
ax1.set_ylabel('Approval rate')
# transform total_quantity to quantile number and graph
quantiles = get_quantile_based_boundaries(train['total_quantity'], 6)
train["total_quantity_binned"] = np.digitize(train['total_quantity'], bins=quantiles)
fig, ax1 = plt.subplots(figsize=(8, 4))
plt.title("total_quantity vs Approval")
sns.pointplot(x="total_quantity_binned", y="project_is_approved", data=train, ci=95, ax=ax1)
ax1.set_ylabel('Approval rate')
# transform mean_cost to quantile number and graph
quantiles = get_quantile_based_boundaries(train['mean_cost'], 6)
train["mean_cost_binned"] = np.digitize(train['mean_cost'], bins=quantiles)
fig, ax1 = plt.subplots(figsize=(8, 4))
plt.title("mean_cost vs Approval")
sns.pointplot(x="mean_cost_binned", y="project_is_approved", data=train, ci=95, ax=ax1)
ax1.set_ylabel('Approval rate')
# transform total_cost to quantile number and graph
quantiles = get_quantile_based_boundaries(train['total_cost'], 6)
train["total_cost_binned"] = np.digitize(train['total_cost'], bins=quantiles)
fig, ax1 = plt.subplots(figsize=(8, 4))
plt.title("total_cost vs Approval")
sns.pointplot(x="total_cost_binned", y="project_is_approved", data=train, ci=95, ax=ax1)
ax1.set_ylabel('Approval rate')
# repeat binning process in test data
bins = [0., 2., 10.,]
test["experience"] = np.digitize(test['teacher_number_of_previously_posted_projects'], bins=bins)

quantiles = get_quantile_based_boundaries(test['unique_items'], 6)
test["unique_items_binned"] = np.digitize(test['unique_items'], bins=quantiles)
quantiles = get_quantile_based_boundaries(test['total_quantity'], 6)
test["total_quantity_binned"] = np.digitize(test['total_quantity'], bins=quantiles)
quantiles = get_quantile_based_boundaries(test['mean_cost'], 6)
test["mean_cost_binned"] = np.digitize(test['mean_cost'], bins=quantiles)
quantiles = get_quantile_based_boundaries(test['total_cost'], 6)
test["total_cost_binned"] = np.digitize(test['total_cost'], bins=quantiles)
# define functions for selected features and target
def preprocess_features(df):
  selected_features = df[
  ["project_subject_categories",
   "project_subject_subcategories",
   "project_title",
   "project_resource_summary",
   "project_essay_1",
   "project_essay_2",
   "experience",
   "unique_items_binned",
   "total_quantity_binned",
   "mean_cost_binned",
   "total_cost_binned",
   "encoded_category",
   "encoded_subcategory"]]
  return selected_features

def preprocess_targets(df):
  output_targets = df["project_is_approved"]
  return output_targets
# define target encode function
def target_encode(trn_series=None, 
                  tst_series=None, 
                  target=None, 
                  min_samples_leaf=1, 
                  smoothing=1,
                  noise_level=0):

    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return ft_trn_series, ft_tst_series
# target encode features
train['encoded_category'], test['encoded_category'] = target_encode(train['project_subject_categories'], test['project_subject_categories'], train['project_is_approved'])
train['encoded_subcategory'], test['encoded_subcategory'] = target_encode(train['project_subject_subcategories'], test['project_subject_subcategories'], train['project_is_approved'])
# randomize train data before splitting validation set
train = train.reindex(
    np.random.permutation(train.index))
# Choose the first n examples for training.
training_examples = preprocess_features(train.head(122080))
training_targets = preprocess_targets(train.head(122080))

# Choose the last n examples for validation.
validation_examples = preprocess_features(train.tail(60000))
validation_targets = preprocess_targets(train.tail(60000))

# Process features for test
test_examples = preprocess_features(test.copy())
# Vectorize subject categories
vectorizer=TfidfVectorizer(stop_words=stop)
vectorizer.fit(train['project_subject_categories'])
train_project_subject_categories = vectorizer.transform(training_examples['project_subject_categories'])
validation_project_subject_categories = vectorizer.transform(validation_examples['project_subject_categories'])
test_project_subject_categories = vectorizer.transform(test_examples['project_subject_categories'])

# Vectorize subject sub-categories
vectorizer.fit(train['project_subject_subcategories'])
train_project_subject_subcategories = vectorizer.transform(training_examples['project_subject_subcategories'])
validation_project_subject_subcategories = vectorizer.transform(validation_examples['project_subject_subcategories'])
test_project_subject_subcategories = vectorizer.transform(test_examples['project_subject_subcategories'])
# Vectorize project title
vectorizer=TfidfVectorizer(stop_words=stop, ngram_range=(1, 2), max_df=0.9, min_df=5, max_features=2000)
vectorizer.fit(train['project_title'])
train_project_title = vectorizer.transform(training_examples['project_title'])
validation_project_title = vectorizer.transform(validation_examples['project_title'])
test_project_title = vectorizer.transform(test_examples['project_title'])

# Vectorize project summary
vectorizer.fit(train['project_resource_summary'])
train_project_resource_summary = vectorizer.transform(training_examples['project_resource_summary'])
validation_project_resource_summary = vectorizer.transform(validation_examples['project_resource_summary'])
test_project_resource_summary = vectorizer.transform(test_examples['project_resource_summary'])
# Vectorize essays
vectorizer=TfidfVectorizer(stop_words=stop, ngram_range=(1, 3), max_df=0.9, min_df=5, max_features=2000)
vectorizer.fit(train['project_essay_1'])
train_project_essay_1 = vectorizer.transform(training_examples['project_essay_1'])
validation_project_essay_1 = vectorizer.transform(validation_examples['project_essay_1'])
test_project_essay_1 = vectorizer.transform(test_examples['project_essay_1'])

vectorizer.fit(train['project_essay_2'])
train_project_essay_2 = vectorizer.transform(training_examples['project_essay_2'])
validation_project_essay_2 = vectorizer.transform(validation_examples['project_essay_2'])
test_project_essay_2 = vectorizer.transform(test_examples['project_essay_2'])
# Define Sentiment Analysis functions

def get_polarity(text):
    textblob = TextBlob(text)
    pol = textblob.sentiment.polarity
    return round(pol,3)

def get_subjectivity(text):
    textblob = TextBlob(text)
    subj = textblob.sentiment.subjectivity
    return round(subj,3)
# Sentiment Analysis in training

training_examples['polarity1'] = training_examples['project_essay_1'].apply(get_polarity)
training_examples['subjectivity1'] = training_examples['project_essay_1'].apply(get_subjectivity)
training_examples['polarity2'] = training_examples['project_essay_2'].apply(get_polarity)
training_examples['subjectivity2'] = training_examples['project_essay_2'].apply(get_subjectivity)
# Sentiment Analysis in validation

validation_examples['polarity1'] = validation_examples['project_essay_1'].apply(get_polarity)
validation_examples['subjectivity1'] = validation_examples['project_essay_1'].apply(get_subjectivity)
validation_examples['polarity2'] = validation_examples['project_essay_2'].apply(get_polarity)
validation_examples['subjectivity2'] = validation_examples['project_essay_2'].apply(get_subjectivity)
# Sentiment Analysis in test

test_examples['polarity1'] = test_examples['project_essay_1'].apply(get_polarity)
test_examples['subjectivity1'] = test_examples['project_essay_1'].apply(get_subjectivity)
test_examples['polarity2'] = test_examples['project_essay_2'].apply(get_polarity)
test_examples['subjectivity2'] = test_examples['project_essay_2'].apply(get_subjectivity)
# Drop unnecessary columns
to_drop = ['project_subject_categories', 'project_subject_subcategories', 'project_title', 'project_essay_1', 'project_essay_2', 'project_resource_summary']
for col in to_drop:
    training_examples.drop([col], axis=1, inplace=True)
    validation_examples.drop([col], axis=1, inplace=True)
    test_examples.drop([col], axis=1, inplace=True)
# Combine all features
training_features = csr_matrix(hstack([training_examples.values, train_project_subject_categories, train_project_subject_subcategories, train_project_title, train_project_resource_summary, train_project_essay_1, train_project_essay_2]))
validation_features = csr_matrix(hstack([validation_examples.values, validation_project_subject_categories, validation_project_subject_subcategories, validation_project_title, validation_project_resource_summary, validation_project_essay_1, validation_project_essay_2]))
test_features = csr_matrix(hstack([test_examples.values, test_project_subject_categories, test_project_subject_subcategories, test_project_title, test_project_resource_summary, test_project_essay_1, test_project_essay_2]))
# XGBoost

#params = {'eta': 0.025, 'max_depth': 16, 'objective': 'binary:logistic', 'eval_metric': 'auc', 'silent': False, 'colsample':0.9}
#watchlist = [(xgb.DMatrix(training_features, training_targets2), 'train'), (xgb.DMatrix(validation_features, validation_targets2), 'valid')]
#model = xgb.train(params, xgb.DMatrix(training_features, training_targets2), 1000,  watchlist, verbose_eval=10, early_stopping_rounds=20)
# LightGBM

params = {
         'boosting_type': 'gbdt',
         'objective': 'binary',
         'metric': 'auc',
         'max_depth': 16,
         'num_leaves': 31,
         'learning_rate': 0.025,
         'feature_fraction': 0.85,
         'bagging_fraction': 0.85,
         'bagging_freq': 5,
         'verbose': 0,
         'num_threads': 1,
         'lambda_l2': 1,
         'min_gain_to_split': 0,
         }  

model2 = lgb.train(
    params,
    lgb.Dataset(training_features, training_targets),
    num_boost_round=10000,
    valid_sets=[lgb.Dataset(validation_features, validation_targets)],
    verbose_eval=100,
    early_stopping_rounds=100
    )
# Submission CSV

submission = pd.DataFrame()
submission['id'] = test['id']
submission['project_is_approved'] = model2.predict(test_features, num_iteration=model2.best_iteration)
submission.to_csv('submission.csv', index=False)