import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.pipeline import Pipeline

from sklearn.pipeline import make_pipeline



from sklearn.linear_model import BayesianRidge

from catboost import CatBoostRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor



# Plot settings

plt.style.use('ggplot')

%config InlineBackend.figure_format = 'svg'
# BERT-as-service

!pip install bert-serving-server

!pip install bert-serving-client
!pip install git+http://github.com/brendanhasz/dsutils.git

    

from dsutils.encoding import LambdaTransformer

from dsutils.encoding import LambdaFeatures

from dsutils.encoding import NullEncoder

from dsutils.encoding import DateEncoder

from dsutils.encoding import JsonEncoder

from dsutils.encoding import NhotEncoder

from dsutils.encoding import JoinTransformer

from dsutils.encoding import MultiTargetEncoderLOO



from dsutils.ensembling import BaggedRegressor

from dsutils.ensembling import StackedRegressor

from dsutils.models import InterpolatingPredictor



from dsutils.evaluation import permutation_importance_cv

from dsutils.evaluation import plot_permutation_importance

from dsutils.evaluation import top_k_permutation_importances



from dsutils.transforms import Scaler, Imputer



from dsutils.cleaning import DeleteCols

from dsutils.cleaning import KeepOnlyCols

from dsutils.external import BertEncoder
# Load training data

dtypes = {

  'id':                    'uint16',

  'belongs_to_collection': 'str',

  'budget':                'float32',

  'genres':                'str',

  'homepage':              'str',

  'imdb_id':               'str',

  'original_language':     'str',

  #'original_title':        'str',

  'overview':              'str',

  'popularity':            'float32',

  #'poster_path':           'str',

  'production_companies':  'str',

  'production_countries':  'str',

  'release_date':          'str',

  'runtime':               'float32',

  'spoken_languages':      'str',

  #'status':                'str',

  'tagline':               'str',

  'title':                 'str',

  'Keywords':              'str',

  'cast':                  'str',

  'crew':                  'str',

  'revenue':               'float32',

}

train = pd.read_csv('../input/tmdb-box-office-prediction/train.csv',

                    usecols=dtypes.keys(),

                    dtype=dtypes)

del dtypes['revenue']

test = pd.read_csv('../input/tmdb-box-office-prediction/test.csv',

                   usecols=dtypes.keys(),

                   dtype=dtypes)

df = pd.concat([train, test], axis=0)
# Load imdb scores

dtypes = {

  'imdb_id':    'str',

  'avg_rating': 'float32',

  'num_rating': 'float32',

}

imdb_df = pd.read_csv('../input/box-office-prediction-imbd-scores/imdb_scores.csv',

                      usecols=dtypes.keys(),

                      dtype=dtypes)
# Histogram of release year (sans century)

year_fn = lambda x: int(x[-2:]) if isinstance(x, str) else np.nan

years = df['release_date'].apply(year_fn)

plt.hist(years, bins=np.arange(100), color='#648FFF')

plt.xlabel('year (sans century)')

plt.ylabel('count')

plt.show()
def fix_dates(date_str):

    if isinstance(date_str, str):

        if int(date_str[-2:]) < 20:

            return date_str[:-2]+'20'+date_str[-2:]

        else:

            return date_str[:-2]+'19'+date_str[-2:]

    else:

        return np.nan

    

# Fix dates

train['release_date'] = train['release_date'].apply(fix_dates)

test['release_date'] = test['release_date'].apply(fix_dates)
plt.hist(df['runtime'], bins=np.arange(250), color='#FFB000')

plt.xlabel('runtime (min)')

plt.ylabel('count')

plt.show()
plt.hist(df['runtime'], bins=np.arange(50), color='#FFB000')

plt.xlabel('runtime (min)')

plt.ylabel('count')

plt.show()
df.loc[df['runtime']==0, :]
# Set runtimes of 0 to nan

train.loc[train['runtime']<1, 'runtime'] = np.nan

test.loc[test['runtime']<1, 'runtime'] = np.nan
# Make id the index

train.set_index('id', inplace=True)

test.set_index('id', inplace=True)
# Split into X and Y

train_y = train['revenue']

train_X = train

del train_X['revenue']
plt.hist(train_y, color='#785EF0')

plt.xlabel('revenue')

plt.ylabel('count')

plt.show()
plt.hist(np.log1p(train_y), color='#785EF0')

plt.xlabel('log(1+revenue)')

plt.ylabel('count')

plt.show()
# Transform target

train_y = np.log1p(train_y)
plt.hist(df['budget'], color='#FE6100')

plt.xlabel('budget')

plt.ylabel('count')

plt.show()
plt.hist(np.log1p(df['budget']), color='#FE6100')

plt.xlabel('log(1+budget)')

plt.ylabel('count')

plt.show()
plt.hist(df['popularity'], color='#DC267F', bins=np.linspace(0, 100, 50))

plt.xlabel('popularity')

plt.ylabel('count')

plt.show()
plt.hist(np.log1p(df['popularity']), color='#DC267F')

plt.xlabel('log(1+popularity)')

plt.ylabel('count')

plt.show()
# Transforms to apply to numeric columns

transforms = {

    'budget': lambda x: np.log1p(x),

    'popularity': lambda x: np.log1p(x),

}



# Columns to null-encode

null_encode_cols = [

    'belongs_to_collection',

    'homepage',

]



# Date encoder

date_cols = {

    'release_date': ('%m/%d/%Y', ['year', 'month', 'day', 'dayofyear', 'dayofweek'])

}



# JSON fields to extract

json_fields = {

    'genres': 'name',

    'production_companies': 'name',

    'production_countries': 'iso_3166_1',

    'spoken_languages': 'iso_639_1',

    'Keywords': 'name',

    'cast': 'name',

    'crew': [('name', 'job', 'Director'),

             ('name', 'job', 'Producer'),

             ('name', 'job', 'Writer'),

             ('name', 'job', 'Editor'),],

}



# Columns to N-hot encode

nhot_cols = [

    'genres_name',

    'original_language',

    'production_countries_iso_3166_1',

    'spoken_languages_iso_639_1',

]



# Columns to target encode

bayesian_c = 5 #regularization

te_cols = [

    'production_companies_name',

    'cast_name',

    'crew_job_Director_name',

    'crew_job_Producer_name',

    'crew_job_Writer_name',

    'crew_job_Editor_name',

]



# Columns to BERT encode

n_pc = 5 #keep top 5 principal components of BERT embeddings

bert_cols = [

    'overview',

    'tagline',

    'title',

    'Keywords_name',

]



# Feature engineering

word_count = lambda e: len(e.split(' ')) if isinstance(e, str) else 0

keyword_count = lambda e: len(e.split(',')) if isinstance(e, str) else 0

new_features = {

    'budget_runtime_ratio': lambda x: x['budget']/x['runtime'],

    'budget_popularity_ratio': lambda x: x['budget']/(x['popularity']+1),

    'budget_year_ratio': lambda x: x['budget']/np.square(x['release_date_year']),

    'popularity_year_ratio': lambda x: x['popularity']/np.square(x['release_date_year']),

    'rating_to_votes_ratio': lambda x: x['avg_rating']/(x['num_rating']+1),

    'runtime_rating_ratio': lambda x: x['runtime']/(x['avg_rating']+1),

    'overview_word_count': lambda x: x['overview'].apply(word_count),

    'tagline_word_count': lambda x: x['tagline'].apply(word_count),

    'keyword_count': lambda x: x['Keywords_name'].apply(keyword_count),

}
# Create the pipeline

preprocessing = Pipeline([

    ('transforms',   LambdaTransformer(transforms)),

    ('join_imbd',    JoinTransformer(imdb_df, 'imdb_id', 'imdb_id')),

    ('null_encoder', NullEncoder(null_encode_cols, delete_old=True)),

    ('date_encoder', DateEncoder(date_cols)),

    ('json_encoder', JsonEncoder(json_fields)),

    ('nhot_encoder', NhotEncoder(nhot_cols, top_n=10)),

    ('add_features', LambdaFeatures(new_features)),

    ('targ_encoder', MultiTargetEncoderLOO(te_cols, bayesian_c=bayesian_c)),

    ('bert_encoder', BertEncoder(bert_cols, n_pc=n_pc)),

    ('scaler',       Scaler()),

    ('imputer',      Imputer()),

])
"""

# Model w/ just catboost + no BERT encoding

model = Pipeline([

    ('transforms',   LambdaTransformer(transforms)),

    ('join_imbd',    JoinTransformer(imdb_df, 'imdb_id', 'imdb_id')),

    ('null_encoder', NullEncoder(null_encode_cols, delete_old=True)),

    ('date_encoder', DateEncoder(date_cols)),

    ('json_encoder', JsonEncoder(json_fields)),

    ('nhot_encoder', NhotEncoder(nhot_cols, top_n=10)),

    ('add_features', LambdaFeatures(new_features)),

    ('targ_encoder', MultiTargetEncoderLOO(te_cols, bayesian_c=bayesian_c)),

    ('delete_cols',  DeleteCols(bert_cols)),

    ('scaler',       Scaler()),

    ('imputer',      Imputer()),

    ('regressor',    CatBoostRegressor(verbose=False))

])

"""



# Model w/ just catboost

model = Pipeline([

    ('preprocessing', preprocessing),

    ('regressor',     CatBoostRegressor(verbose=False))

])



# Fit + make predictions

fit_model = model.fit(train_X, train_y)

preds = fit_model.predict(test)



# Save predictions to file

preds_df = pd.DataFrame(index=test.index)

preds_df['revenue'] = np.maximum(0, np.expm1(preds))

preds_df.to_csv('predictions_baseline.csv')
# Preprocess both test and train data

train_pp = preprocessing.fit_transform(train_X, train_y)

test_pp = preprocessing.transform(test)
train_pp
%%time



# CatBoost

catboost = Pipeline([

    ('regressor', CatBoostRegressor(verbose=False))

])



# Compute permutation-based feature importances

imp_df = permutation_importance_cv(train_pp,

                                   train_y.copy(),

                                   estimator=catboost,

                                   metric='rmse',

                                   n_jobs=1)
# Show the feature importances

plt.figure(figsize=(6, 20))

plot_permutation_importance(imp_df)

plt.show()
# Get a list of the top 30 most important features

cols_to_keep = top_k_permutation_importances(imp_df, k=30)
# Add feature selection to preprocessing

preprocessing = Pipeline([

    ('preprocessing', preprocessing),

    ('col_filter',    KeepOnlyCols(cols_to_keep)),

])
# Base learner models

base_learners = [

    BayesianRidge(),

    XGBRegressor(),

    CatBoostRegressor(verbose=False),

    LGBMRegressor()

]



# Stacked model

model = StackedRegressor(base_learners,

                         meta_learner=BayesianRidge(),

                         preprocessing=preprocessing,

                         n_splits=5, n_jobs=1)
%%time



# Fit model and predict on test data

fit_model = model.fit(train_X, train_y)

preds = fit_model.predict(test)



# Save predictions to file

preds_df = pd.DataFrame(index=test.index)

preds_df['revenue'] = np.maximum(0, np.expm1(preds))

preds_df.to_csv('predictions.csv')