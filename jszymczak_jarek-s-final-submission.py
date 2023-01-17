import numpy as np

import pandas as pd

import lightgbm as lgb

from sklearn.base import TransformerMixin, BaseEstimator

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.feature_extraction import DictVectorizer



import time

import os



from sklearn.feature_extraction.text import HashingVectorizer

# allowing FeatureUnion to use the scratch surface

os.environ['JOBLIB_TEMP_FOLDER'] = '.'
train_df = pd.read_csv('/kaggle/input/tryinclass/train.csv', parse_dates=['posted_date', 'expiration_date', 'teacher_first_project_date'])

test_df = pd.read_csv('/kaggle/input/tryinclass/test.csv', parse_dates=['posted_date', 'expiration_date', 'teacher_first_project_date'])
# based on poor man's calculation (a little bit wrong one) and guestimation for 2018

budgets ={    

    2013: 4.283420e+07,

    2014: 6.440082e+07,

    2015: 7.252042e+07,

    2016: 9.196045e+07,

    2017: 9.196045e+07,

    2018: 10.000000e+07

}



class CustomFeatures(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):

        return self



    def transform(self, df):

        df['project_lifespan'] = (df['expiration_date'] - df['posted_date']).dt.total_seconds() / (3600*24.0)

        df['teacher_experience'] = (df['posted_date'] - df['teacher_first_project_date']).dt.total_seconds() / (3600*24.0)

        df['teacher_activity'] = df['teachers_project_no'] / (df['teacher_experience'] / 365.0)

        total_text_len = 0

        for tc in text_columns:

            str_len = df[tc].str.len()

            total_text_len += str_len

            df['{}_len'.format(tc)] = str_len

        df['total_text_len'] = total_text_len

        df['budget'] = df.posted_date.dt.year.apply(lambda v: budgets[v])

        df['adjusted_cost'] = df['cost'] / df['budget']

        # datasets are sorted, normally transformer should ensure that

        for year in df.posted_date.dt.year.unique():

            df.loc[df.posted_date.dt.year == year, 'cumulative_adjusted_cost'] = df.loc[df.posted_date.dt.year == year].adjusted_cost.cumsum()

        return df



class ItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):

        self.key = key



    def fit(self, x, y=None):

        return self



    def transform(self, data_dict):

        return data_dict[self.key]



class FillNa(BaseEstimator, TransformerMixin):

    def __init__(self, value):

        self.value = value



    def fit(self, x, y=None):

        return self



    def transform(self, X):

        return X.fillna(self.value)

    

class LoggingTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, message):

        self.message = message



    def fit(self, X, y=None):

        print("{}: {}, dataset shape: {}".format(time.strftime("%Y-%m-%d %H:%M:%S",

                                                               time.gmtime()), self.message, X.shape))

        return self



    def transform(self, X):

        return X

    

class DF2Dict(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):

        return self



    def transform(self, df):

        return df.to_dict('records')
dv_columns = ['school_id', 'teacher_id', 'teachers_project_no', 'project_type', 'project_category',

              'project_subcategory', 'project_grade_level', 'resource_category', 'adjusted_cost', 'cumulative_adjusted_cost',

               'teacher_gender', 'metro_type', 'free_lunch_percentage', 'state', 'city', 'county',

               'district']



date_features = ['posted_date', 'expiration_date', 'teacher_first_project_date']



text_columns = ['project_title', 'project_essay', 'project_description', 'project_statement']



extra_features = ['project_lifespan', 'teacher_experience', 'teacher_activity']



text_len_columns = ['project_title_len', 'project_essay_len', 'project_description_len', 'project_statement_len', 'total_text_len']

pipeline = lambda: Pipeline([

    ('log1', LoggingTransformer("Starting feature extraction pipeline")),

    ('custom_features', CustomFeatures()),

    ('log2', LoggingTransformer("Processed custom features, moving to pipeline")),    

    ('feature_union', FeatureUnion([

        ('dv_features', Pipeline([

            ('log1', LoggingTransformer("Applying dict vectorizer")),            

            ('selector', ItemSelector(dv_columns + extra_features + text_len_columns)),

            ('df2dict', DF2Dict()),

            ('vectorizer', DictVectorizer()),

            ('log2', LoggingTransformer("End of dict vectorizer"))

        ])),            

        ('text_features0', Pipeline([

            ('log1', LoggingTransformer("Applying hashing vectorizer")),

            ('selector', ItemSelector(text_columns[0])),            

            ('fillna', FillNa('')),      

            ('vectorizer',

             HashingVectorizer(stop_words='english', alternate_sign=False, analyzer='word', ngram_range=(1, 1), strip_accents='ascii')),

            ('log2', LoggingTransformer("End of hashing vectorizer")),

        ])),

        ('text_features1', Pipeline([

            ('log1', LoggingTransformer("Applying hashing vectorizer")),

            ('selector', ItemSelector(text_columns[1])),            

            ('fillna', FillNa('')),        

            ('vectorizer',

             HashingVectorizer(stop_words='english', alternate_sign=False, analyzer='word', ngram_range=(1, 1), strip_accents='ascii')),

            ('log2', LoggingTransformer("End of hashing vectorizer")),

        ])),

        ('text_features2', Pipeline([

            ('log1', LoggingTransformer("Applying hashing vectorizer")),

            ('selector', ItemSelector(text_columns[2])),            

            ('fillna', FillNa('')),     

            ('vectorizer',

             HashingVectorizer(stop_words='english', alternate_sign=False, analyzer='word', ngram_range=(1, 1), strip_accents='ascii')),

            ('log2', LoggingTransformer("End of hashing vectorizer")),

        ])),

        ('text_features3', Pipeline([

            ('log1', LoggingTransformer("Applying hashing vectorizer")),

            ('selector', ItemSelector(text_columns[3])),            

            ('fillna', FillNa('')),        

            ('vectorizer',

             HashingVectorizer(stop_words='english', alternate_sign=False, analyzer='word', ngram_range=(1, 1), strip_accents='ascii')),

            ('log2', LoggingTransformer("End of hashing vectorizer")),

        ])),        

    ], n_jobs=1)),

    ('log3', LoggingTransformer("Done")),

]

)

pp = pipeline()

y = train_df['funded_or_not']

X = pp.fit_transform(train_df)

X_test = pp.transform(test_df)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)



clf = lgb.LGBMClassifier(

    num_leaves=80, 

    max_depth=-1, 

    random_state=42, 

    n_jobs=-1, 

    n_estimators=1000,

    colsample_bytree=0.7,

    subsample=0.7,

    learning_rate=0.05

)



clf.fit(X_train, y_train,

        early_stopping_rounds=10, 

        eval_metric = 'auc', 

        eval_set = [(X_train, y_train), (X_val, y_val)],

        eval_names = ['train', 'valid'],

        verbose = 10

)
threshold = 0.5

y_test_proba = clf.predict_proba(X_test)[:,1]

y_test = np.where(y_test_proba > threshold, 1, 0)

submission = test_df[['project_id']].copy()

submission['funded_or_not'] = y_test

submission.to_csv('submission.csv', index=False)