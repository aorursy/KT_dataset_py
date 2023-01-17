# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')



train_df.head()
train_df.describe()
train_df.describe(include=['O'])
train_df.describe(include=['O'])
train_df["Sex"].unique()
train_df["Embarked"].unique()
train_df["Sex"].unique()
train_df["Embarked"].unique()
train_df.info()
train_df.hist(bins=20, figsize=(12,7))

plt.show()
corr_matrix = train_df.corr()

corr_matrix["Survived"].sort_values(ascending=True)
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['Sex','Survived']].groupby(['Sex'], as_index=False).mean()
train_df[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean()
train_df[['Parch','Survived']].groupby(['Parch'],as_index=False).mean()
from sklearn.preprocessing import Imputer,LabelBinarizer, StandardScaler

from sklearn.pipeline import Pipeline, FeatureUnion, BaseEstimator, TransformerMixin



num_attribs = ["Age","SibSp","Parch","Fare"]

cat_attribs = ["Pclass","Sex","Embarked"]





class DataFrameSelector(TransformerMixin):

    def __init__(self, attribute_names):

        """

        

        """        

        self.attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names]



class CategoryValueImputer(TransformerMixin):

    def __init__(self):

        """

        

        """

        self.fill_with = {}

    def fit(self, X, y=None):

        for c in X:

            self.fill_with[c] = X[c].value_counts().index[0]

        return self

    def transform(self, X):

        return X.fillna(self.fill_with)



class LabelBinarizerWrapper(TransformerMixin):

    def __init__(self):

        """

        """

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        for c in X:

            encoder = LabelBinarizer() # Optimize: OHV for Pclass/Embarked?

            encoded = encoder.fit_transform(X[c])

            X[c] = encoded

        return X



num_pipeline = Pipeline([

    ('selector', DataFrameSelector(num_attribs)),    

    ('imputer', Imputer(strategy="median")), # OPTIMIZE: better way to fill missing Age values

    ('std_scaler', StandardScaler())

])

    

cat_pipeline = Pipeline([

    ('selector', DataFrameSelector(cat_attribs)),

    ('imputer', CategoryValueImputer()), # OPTIMIZE: better way to fill missing Embarked values

    ('label_binarizer', LabelBinarizerWrapper())

])



combined_pipeline = FeatureUnion(transformer_list=[

    ('num_pipeline', num_pipeline),

    ('cat_pipeline', cat_pipeline)

])

train_df_num = combined_pipeline.fit_transform(train_df)

train_df_num
from sklearn.preprocessing import Imputer,LabelBinarizer, StandardScaler

from sklearn.pipeline import Pipeline, FeatureUnion, BaseEstimator, TransformerMixin



num_attribs = ["Age","SibSp","Parch","Fare"]

cat_attribs = ["Pclass","Sex","Embarked"]





class DataFrameSelector(TransformerMixin):

    def __init__(self, attribute_names):

        """

        

        """        

        self.attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names].values



class CategoryValueImputer(TransformerMixin):

    def __init__(self):

        """

        

        """        

    def fit(self, X, y=None):

        #if X.dtype == np.dtype('O'):

        #    self.fill_with = X.value_counts().index[0]

        #else:

        #    self.fill_with = X.mean()

        #return self

        print(X)

        self.fill_with = pd.Series([X[c].value_counts().index[0]

            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],

            index=X.columns)

        

    

    def transform(self, X):

        return X.fillna(self.fill_with)





num_pipeline = Pipeline([

    ('selector', DataFrameSelector(num_attribs)),    

    ('imputer', Imputer(strategy="median")), # OPTIMIZE: better way to fill missing Age values

    ('std_scaler', StandardScaler())

])

    

cat_pipeline = Pipeline([

    ('selector', DataFrameSelector(cat_attribs)),

    ('imputer', CategoryValueImputer()), # OPTIMIZE: better way to fill missing Embarked values

    ('label_binarizer', LabelBinarizer())

])



combined_pipeline = FeatureUnion(transformer_list=[

    ('num_pipeline', num_pipeline),

    ('cat_pipeline', cat_pipeline)

])

#train_df_num = combined_pipeline.fit_transform(train_df)

#train_df_num



#imputer = CategoryValueImputer()

#train_df[cat_attribs]["Sex"].value_counts()

#cat_pipeline.fit_transform(train_df)

cat_array = train_df[cat_attribs]

for c in cat_array:

  cat_array[c].value_counts()
lb = LabelBinarizer()

#lb.fit_transform(train_df["Embarked"])

embarked_full = train_df["Embarked"].dropna()

lb.fit_transform(embarked_full)