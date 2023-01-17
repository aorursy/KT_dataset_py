# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re



from sklearn.base import TransformerMixin, BaseEstimator

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split



import tensorflow as tf



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
class AdditionalFeatures(BaseEstimator, TransformerMixin):

    """ 

    Transformer that extracts new features from the Titanic dataset, these being

    family_size, is_alone, has_cabin and title

    """

    

    def fit(self, x, y=None):

        return self

    

    def transform(self, x, y=None):

        x = x.copy()

        

        # new attribute family size, essentially integrates Parch with SibSp

        x['family_size'] = x['Parch'] + x['SibSp']



        # categorical attribute is alone, indicates whether the person traveled alone or not

        x['is_alone'] = x.apply(lambda row: True if row.family_size == 0 else False, axis=1)



        # categorical feature that indicates if the person has a cabin or not

        x['has_cabin'] = x.Cabin.apply(lambda row: False if pd.isnull(row) else True)



        # extract title from the people's names and map odd titles to correct ones

        x['title'] = x['Name'].apply(lambda name: re.search(r'(\w+)\.', name).groups()[0])

        title_mapping = {'Mme': 'Mrs', 'Mlle' : 'Miss', 'Major':'Mr', 'Col':'Mr', 'Capt':'Mr', 'Don':'Mr', 'Lady':'Mrs', 'Ms':'Miss', 'Countess':'Mrs', 'Sir':'Mr',

                        'Jonkheer':'Mr', 'Dona':'Mrs',}



        x.replace({'title': title_mapping}, inplace=True)



        # simple conversion of sex into a boolean value instead

        x['is_male'] = x.Sex.apply(lambda row: True if row == 'male' else False)

        

        return x
class CategoryImputer(BaseEstimator, TransformerMixin):

    """ Given an attribute and a categorical attribute, impute

    the provided attribute using aggregate functions calculated

    from the those within the category 

    """

    

    def __init__(self, impute_attr, category, strategy='median'):

        self.impute_attr = impute_attr

        self.category = category

        self.strategy = strategy

        

    def fit(self, x, y=None):

        return self

    

    def transform(self, x, y=None):

        age_by_cat = dict(x.groupby(by=self.category)[self.impute_attr].agg(self.strategy))

        impute_values = x[self.category].apply(lambda i: age_by_cat[i])

        x[self.impute_attr].fillna(impute_values, inplace=True)

        return x
class AttributeSelector(BaseEstimator, TransformerMixin):

    """

    Helper transformer that returns only a subset of the features

    """

    def __init__(self, feature_names):

        self.feature_names = feature_names

    

    def fit(self, x, y=None):

        return self

    

    def transform(self, x, y=None):

        return x[self.feature_names]
train_data = pd.read_csv('/kaggle/input/titanic/train.csv', index_col='PassengerId')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv', index_col='PassengerId')



# split the labels and data

x_train = train_data.drop('Survived', axis=1)

y_train = train_data['Survived'].copy()

x_test = test_data
# names of categorical and numerical features

num_feats = ['Age', 'SibSp', 'Parch', 'Fare', 'family_size']

cat_feats = ['Pclass', 'Sex', 'title', 'Embarked', 'has_cabin', 'is_alone']
# pipeline for numerical variables with the following sequence:

# adds new features (family_size, is_alone, has_cabin and title)

# Impute the age using medians of the categorical title class

# Impute the fare using medians based on the PClass

# Select only the numerical features and impute with the median values

# Standardize the features

num_pipeline = Pipeline([

            ('new_feats', AdditionalFeatures()),

            ('age_impute', CategoryImputer('Age', 'title')),

            ('fare_impute', CategoryImputer('Fare', 'Pclass')),

            ('num_select', AttributeSelector(num_feats)),

            ('imputer', SimpleImputer(strategy='median')),

            ('scaler', StandardScaler()),

])



# pipeline for the categorical variables with the following sequence:

# add new features (family_size, is_alone, has_cabin and title)

# select only the categorical variables, and impute with the most frequent value

# apply one-hot-encoding

cat_pipeline = Pipeline([

            ('new_feats', AdditionalFeatures()),

            ('cat_select', AttributeSelector(cat_feats)),

            ('mode_imputer', SimpleImputer(strategy="most_frequent")),

            ('one-hot', OneHotEncoder())

])



# merged pipeline without the estimator

pipeline_no_est = Pipeline([

    ('data_prep', FeatureUnion([

        ('num_prep', num_pipeline),

        ('cat_prep', cat_pipeline),

    ])),

])
# apply the created pipeline to the data

x_train_pre = pipeline_no_est.fit_transform(x_train)



# create neural network

model = tf.keras.models.Sequential([

    tf.keras.layers.Input(shape=(x_train_pre.shape[1])),

    tf.keras.layers.Dense(1024, activation=tf.nn.relu),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(1024, activation=tf.nn.relu),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)

])





model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])
# fit the data to the model

model.fit(x_train_pre, y_train, epochs=300)
# prepare test data for prediction

x_test_pre = pipeline_no_est.transform(x_test)

x_test['Survived'] = model.predict_classes(x_test_pre)

preds = pd.DataFrame(x_test['Survived'].copy())



# save to submition file

preds.to_csv('submission.csv')