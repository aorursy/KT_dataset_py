import category_encoders as ce

import numpy as np

import pandas as pd

import sklearn as sk

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from sklearn.pipeline import Pipeline

from sklearn.model_selection import KFold, GridSearchCV

from sklearn.base import TransformerMixin

from sklearn.compose import ColumnTransformer

import warnings
warnings.simplefilter('ignore')
TRAIN_PATH = '/kaggle/input/cat-in-the-dat/train.csv'

TEST_PATH = '/kaggle/input/cat-in-the-dat/test.csv'



train = pd.read_csv(TRAIN_PATH)

test = pd.read_csv(TEST_PATH)

train.describe()
y = train.pop('target')

y.head()
class AdjustDay(TransformerMixin):

    """

    Noticed from EDA that there is a very simple symmetric pattern that we can encode day as an ordinal category. 

    It turns out this didn't improve the score, but I'll leave it in as a nice touch (or for people wanting to know 

    how to build a basic sklearn custom transformer). ;)

    """



    def fit(self, X, y=None):

        return self



    def transform(self, X):

        return abs(X-4)
#One Hot Encoded stuff

ohe_step = ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))

ohe_pipeline = Pipeline(steps=[ohe_step])

cat_cols = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'month']



#Catboost Encoded stuff

catboost_enc_step = ('catboost', ce.CatBoostEncoder())

catboost_pipeline = Pipeline(steps=[catboost_enc_step])

high_dim_cols = ['nom_7', 'nom_8', 'nom_9']



#Ordinal Encoded automatically

ord_auto_step = ('ord', OrdinalEncoder(categories='auto'))

ord_auto_pipeline = Pipeline(steps=[ord_auto_step])

ord_auto_cols = ['ord_0', 'ord_3', 'ord_4', 'ord_5']



#Ordinal Encoded with manually specified order

ord_list_step = ('ord_list', OrdinalEncoder(categories=[['Novice', 'Contributor', 'Expert', 'Master', 'Grandmaster'],

                                                    ['Freezing', 'Cold', 'Warm', 'Hot', 'Boiling Hot', 'Lava Hot']]))

ord_list_pipeline = Pipeline(steps=[ord_list_step])

ord_list_cols = ['ord_1', 'ord_2']



#Using custom transformer

adjust_day = ('adjust_day', AdjustDay())

adjust_day_pipeline = Pipeline(steps=[adjust_day, ord_auto_step])

day_cols = ['day']



#All the transformers

transformers = [('cat', ohe_pipeline, cat_cols), 

                ('ord_auto', ord_auto_pipeline, ord_auto_cols), 

                ('day', adjust_day_pipeline, day_cols),

                ('ord_list', ord_list_pipeline, ord_list_cols),

                ('high_dim_cols', catboost_pipeline, high_dim_cols)]

col_transformer = ColumnTransformer(transformers=transformers)



#Create the pipeline

ml_pipe = Pipeline([('transform', col_transformer), ('lr', LogisticRegression())])
#Cross-Validation

kf = KFold(n_splits=5, shuffle=True, random_state=123)

param_grid = {

    'lr__C': [.001, 0.01, 0.1, 1.0, 10.0],

    }

gs = GridSearchCV(ml_pipe, param_grid, cv=kf)

gs.fit(train, y)

print(gs.best_params_)

print(gs.best_score_)