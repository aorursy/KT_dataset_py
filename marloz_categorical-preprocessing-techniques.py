import pandas as pd

import numpy as np
df = pd.read_csv('../input/home-credit-default-risk/application_train.csv')
df.shape
df.dtypes.value_counts()
df.head()
import pylab

import matplotlib.pyplot as plt



def plot_missing_values(df):

    """ For each column with missing values plot proportion that is missing."""

    data = [(col, df[col].isnull().sum() / len(df)) 

            for col in df.columns if df[col].isnull().sum() > 0]

    col_names = ['column', 'percent_missing']

    missing_df = pd.DataFrame(data, columns=col_names).sort_values('percent_missing')

    pylab.rcParams['figure.figsize'] = (15, 8)

    missing_df.plot(kind='barh', x='column', y='percent_missing'); 

    plt.title('Percent of missing values in colummns');
plot_missing_values(df)
f'Total missing values: {df.isnull().sum().sum()}'
df.TARGET.value_counts()
def downsample_df(df, target_col):

    positive_class = df[df[target_col] == 1]

    negative_class = (df[df[target_col] == 0]

                      .sample(n=positive_class.shape[0],

                              random_state=42))

    return pd.concat([positive_class, negative_class], axis=0)
sampled_df = downsample_df(df, 'TARGET')
print(sampled_df.shape)

sampled_df.TARGET.value_counts()
from sklearn.model_selection import train_test_split
y = sampled_df.TARGET

X = sampled_df.drop('TARGET', axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=.2, random_state=42)

X_train.shape, X_valid.shape, y_train.shape, y_valid.shape
def get_categorical_columns(X):

    return [col for col in X.columns if X[col].dtypes == 'O']
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder



cat_imputer = SimpleImputer(strategy='constant', fill_value='none')

ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

cat_processor = Pipeline([

    ('imputer', cat_imputer),

    ('encoder', ohe)

])



num_imputer = SimpleImputer(strategy='median')



preprocessor = ColumnTransformer([('categoricals', cat_processor, get_categorical_columns)],

                                 remainder=num_imputer)
_ = pd.DataFrame(preprocessor.fit_transform(X_train))

print(f'Number of missing values after imputatation {_.isnull().sum().sum()}')

print(f'All data types are numeric: {sum(_.dtypes == float) == _.shape[1]}')
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(random_state=42, n_jobs=-1)

pipeline = Pipeline([

    ('preprocessor', preprocessor),

    ('classifier', rf)

])
pipeline.fit(X_train, y_train)

pipeline.score(X_valid, y_valid)
min_count_categories = [(col, X_train[col].value_counts().min()) for col in get_categorical_columns(X_train)]

pd.DataFrame(min_count_categories).set_index(0)[1].sort_values().plot(kind='bar')

plt.yscale('log')
unique_counts = [(col, X_train[col].nunique()) for col in get_categorical_columns(X_train)]

pd.DataFrame(unique_counts).set_index(0)[1].sort_values().plot(kind='bar')
def combine_low_count_classes(X, threshold=100):

    """ This is a transformer function inteded to be used on categorical columns without any missing values.

    It loops through variables and checks for categories that have related counts lower than specified threshold.

    Then it combines all these low count categories into single 'other' category,

    along with first category above the threshold"""

    X = pd.DataFrame(X)

    for column in X.columns:

        frequencies = X[column].value_counts()

        if frequencies.min() < threshold:

            lower_bound = frequencies[frequencies >= threshold].min()

            mask = frequencies[frequencies <= lower_bound].index

            replace_dict = dict.fromkeys(mask, 'other')

            X[column] = X[column].replace(replace_dict)

    return X.values
from sklearn.preprocessing import FunctionTransformer



combiner = FunctionTransformer(combine_low_count_classes)
cat_processor = Pipeline([

    ('imputer', cat_imputer),

    ('combiner', combiner),

    ('encoder', ohe)

])

pipeline.set_params(**{'preprocessor__categoricals': cat_processor});
pipeline.fit(X_train, y_train)

pipeline.score(X_valid, y_valid)
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import OrdinalEncoder



class CustomLabelEncoder(BaseEstimator, TransformerMixin):

    """ Fits sklearn's OrdinalEncoder (OE) to categorical columns without missing values.

    Loops through columns and checks if each category is in within fitted OE categories.

    If not, new category is assigned 'new_category' value and appended to OE categories,

    such that OE can be applied to unseen data."""

    

    def fit(self, X, y=None):

        self.oe = OrdinalEncoder()

        self.oe.fit(X)

        return self

    

    def transform(self, X):

        for col_idx in range(X.shape[1]):

            X[:, col_idx] = self.replace_new_categories(X[:, col_idx], self.oe.categories_[col_idx])

            self.oe.categories_[col_idx] = np.append(self.oe.categories_[col_idx], 'new_category')

        return self.oe.transform(X)

    

    def replace_new_categories(self, col, categories):

        return pd.Series(col).map(lambda current_category: 'new_category' 

                                  if current_category not in categories 

                                  else current_category).values
cat_processor = Pipeline([

    ('imputer', cat_imputer),

    ('encoder', CustomLabelEncoder())

])

pipeline.set_params(**{'preprocessor__categoricals': cat_processor});
pipeline.fit(X_train, y_train)

pipeline.score(X_valid, y_valid)
class TargetEncoder(BaseEstimator, TransformerMixin):

    """ Calculates global mean on train set, then proceeds with creating encoding dictionary

    for each column, which contains smoothed target variable mean for each category.

    The smoothing parameter can be changed upon initiation of the class and represents

    number of values per category, i.e. for rare categories global mean has more weight.

    

    Transform part uses same trick as customized Ordinal Encoder used above - 

    checking for new categories in data and assigning them global mean.

    """

    

    def __init__(self, smooth_weight=100):

        self.smooth_weight = smooth_weight

    

    def fit(self, X, y):

        self.global_mean = np.mean(y)

        self.enc_dict = self.create_encoding_dict(X, y)

        return self

    

    def create_encoding_dict(self, X, y):

        enc_dict = {}

        for col_idx in range(X.shape[1]):

            enc_dict[col_idx] = self.get_smooth_means_for_col(X[:, col_idx], y)

        return enc_dict

    

    def get_smooth_means_for_col(self, col, y):

        smooth_mean_agg = (lambda x: (x['count'] * x['mean'] + self.smooth_weight * self.global_mean) 

                           / (x['count'] + self.smooth_weight))

        col_y_concat = pd.concat([pd.Series(col, name='col'), pd.Series(y, name='target')], axis=1)

        return (col_y_concat.groupby('col')['target'].agg(['count', 'mean'])

               .assign(smooth_mean=smooth_mean_agg))['smooth_mean'].to_dict()

    

    def transform(self, X):

        for col_idx in range(X.shape[1]):

            X[:, col_idx] = self.replace_new_categories(X[:, col_idx], self.enc_dict[col_idx].keys())

            X[:, col_idx] = (pd.Series(X[:, col_idx]).map(self.enc_dict[col_idx])

                             .fillna(self.global_mean).values)

        return X

    

    def replace_new_categories(self, col, categories):

        return pd.Series(col).map(lambda current_category: 'new_category' 

                                  if current_category not in categories 

                                  else current_category).values
cat_processor = Pipeline([

    ('imputer', cat_imputer),

    ('encoder', TargetEncoder())

])

pipeline.set_params(**{'preprocessor__categoricals': cat_processor});
pipeline.fit(X_train, y_train)

pipeline.score(X_valid, y_valid)