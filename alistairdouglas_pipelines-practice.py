import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.options.display.max_rows = 999



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
X_train = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv', index_col="Id")

X_test = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv', index_col="Id")
X_train.drop(X_train[(X_train['GrLivArea'] > 4000) & (X_train['SalePrice'] < 250000)].index, inplace=True)



# Log the target value and set y = target

X_train['SalePrice'] = np.log1p(X_train['SalePrice'])

y = X_train['SalePrice']

X_train.drop(axis=1, columns=['SalePrice'], inplace=True)
X_train['MSSubClass'] = X_train['MSSubClass'].astype(str)

X_test['MSSubClass'] = X_test['MSSubClass'].astype(str)



X_train['YearBuilt'] = X_train['YearBuilt'].astype(str)

X_test['YearBuilt'] = X_test['YearBuilt'].astype(str)



X_train['GarageYrBlt'] = X_train['GarageYrBlt'].astype(str)

X_test['GarageYrBlt'] = X_test['GarageYrBlt'].astype(str)



X_train['YearRemodAdd'] = X_train['YearRemodAdd'].astype(str)

X_test['YearRemodAdd'] = X_test['YearRemodAdd'].astype(str)



X_train['MoSold'] = X_train['MoSold'].astype(str)

X_test['MoSold'] = X_test['MoSold'].astype(str)



X_train['YrSold'] = X_train['YrSold'].astype(str)

X_test['YrSold'] = X_test['YrSold'].astype(str)
import itertools

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import FunctionTransformer

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

import category_encoders as ce

from sklearn.base import BaseEstimator, TransformerMixin



class Drop_Features(BaseEstimator, TransformerMixin):

    def __init__(self, drop_threshold=0.95):

        self.drop_threshold = drop_threshold

    

    def fit(self, X, y=None):

        self.cols_to_drop = [cname for cname in X.columns if X[cname].isnull().sum() >= self.drop_threshold*X.shape[0]]

        return self

    

    def transform(self, X):

        X_copy = X.copy()

        X_copy.drop(axis=1, columns=self.cols_to_drop, inplace=True)

        return X_copy



# Create interactions between object columns

class Create_Interactions(BaseEstimator, TransformerMixin):

    def __init__(self):

        return

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X):

        cat_cols = [cname for cname in X.columns if X[cname].dtype == 'object']

        

        interactions = pd.DataFrame(index=X.index)

        

        for cname1, cname2 in itertools.combinations(cat_cols, 2):

            new_cname = "_".join([cname1, cname2])

            new_col = X[cname1].map(str) + "_" + X[cname2].map(str)

            interactions[new_cname] = LabelEncoder().fit_transform(new_col)

        

        X_copy = pd.concat([X, interactions], axis=1)

        

        return X_copy 

    

# Find total number of bathrooms

class Add_Features(BaseEstimator, TransformerMixin):

    def __init__(self, add_feature=True):

        self.add_feature = add_feature

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X):

        if self.add_feature:

            total_bathrooms = X['FullBath'] + X['BsmtFullBath'] + 0.5*(X['BsmtHalfBath']+X['HalfBath'])

            total_area = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']

            

            new_features = pd.DataFrame(np.c_[total_bathrooms, total_area], dtype='float64', columns=['total_bathrooms', 'total_area'], index=X.index)

            X_copy = pd.concat([X, new_features], axis=1)

            return X_copy

        else:

            return X

    

class Skewed_Features(BaseEstimator, TransformerMixin):

    def __init__(self, skew_threshold=0.8):

        self.skew_threshold = skew_threshold

    

    def fit(self, X, y=None):

        skewed_cols = X.select_dtypes(include=[np.number]).skew()

        self.top_skewed_cols = skewed_cols[abs(skewed_cols > self.skew_threshold)].index

        return self

    

    def transform(self, X):

        # Make a copy so that X is not changed

        X_copy = X.copy()

        

        for cname in self.top_skewed_cols:

            X_copy[cname] = np.log1p(X_copy[cname])

        return X_copy

        

class Colinear_Features(BaseEstimator, TransformerMixin):

    def __init__(self, corr_threshold=0.8):

        self.corr_threshold = corr_threshold

    

    def fit(self, X, y=None):

        num_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]

        

        feature_correllation = X.corr(method='kendall')

        # iterate over rows

        columns = np.full((feature_correllation.shape[0],), True, dtype=bool)

        for i in range(feature_correllation.shape[0]):

            for j in range(i+1, feature_correllation.shape[0]):

                if feature_correllation.iloc[i,j] >= self.corr_threshold:

                    if columns[j]:

                        columns[j] = False

        self.cols_to_drop = list(set(num_cols) - set(feature_correllation.columns[columns]))

        return self

    

    def transform(self, X):

        # Make a copy so that X is not changed

        X_copy = X.copy()

        X_copy.drop(axis=1, columns=self.cols_to_drop, inplace=True)

        return X_copy

    

class Imputer(BaseEstimator, TransformerMixin):

    def __init__(self):

        return

    

    def fit(self, X, y=None):

        self.num_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]

        self.cat_cols = [cname for cname in X.columns if X[cname].dtype == 'object']

        self.zero_imputer = SimpleImputer(strategy='constant', fill_value=0.0)

        self.medi_imputer = SimpleImputer(strategy='median')

        self.none_imputer = SimpleImputer(strategy='constant', fill_value='none')

        self.freq_imputer = SimpleImputer(strategy='most_frequent')

        

        cols_to_clean = {}



        # Numeric data

        cols_to_clean['num_cols_zeros'], cols_to_clean['num_cols_median'], cols_to_clean['num_cols_skew'] = [],[],[]



        for column in self.num_cols:

            if X[column].min() == 0:

                cols_to_clean['num_cols_zeros'].append(column)

            else:

                cols_to_clean['num_cols_median'].append(column)

        

        # Categorical data

        cols_to_clean['cat_cols_freq'], cols_to_clean['cat_cols_none'] = [], []



        for column in self.cat_cols:

            if X[column].isnull().sum() <= 30:

                cols_to_clean['cat_cols_freq'].append(column)

            else:

                cols_to_clean['cat_cols_none'].append(column)

        

        # Fit all of the transforms

        for cname in cols_to_clean['num_cols_zeros']:

            self.zero_imputer.fit(X[[cname]])

            

        for cname in cols_to_clean['num_cols_median']:

            self.medi_imputer.fit(X[[cname]])

        

        for cname in cols_to_clean['cat_cols_none']:

            self.none_imputer.fit(X[[cname]])

            

        for cname in cols_to_clean['cat_cols_freq']:

            self.freq_imputer.fit(X[[cname]])

        

        self.cols_to_clean = cols_to_clean

        return self

    

    def transform(self, X):

        

        # Make a copy so that X is not changed

        X_copy = X.copy()

        

        # Apply transformations

        for cname in self.cols_to_clean['num_cols_zeros']:

            X_copy[cname] = self.zero_imputer.transform(X[[cname]])

            

        for cname in self.cols_to_clean['num_cols_median']:

            X_copy[cname] = self.medi_imputer.transform(X[[cname]])

        

        for cname in self.cols_to_clean['cat_cols_none']:

            X_copy[cname] = self.none_imputer.transform(X[[cname]])

            

        for cname in self.cols_to_clean['cat_cols_freq']:

            X_copy[cname] = self.freq_imputer.transform(X[[cname]])

            

        return X_copy

    

class Encoder(BaseEstimator, TransformerMixin):

    def __init__(self):

        return

    

    def fit(self, X, y=None):

        self.num_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]

        self.cat_cols = [cname for cname in X.columns if X[cname].dtype == 'object']

        

        self.count_encoder = ce.CountEncoder()

        self.target_encoder = ce.TargetEncoder(cols=self.cat_cols)

        self.OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

        

        self.count_encoder.fit(X[self.cat_cols])

        self.target_encoder.fit(X[self.cat_cols], y)

        self.OH_encoder.fit(X[self.cat_cols])

        

        return self

    

    def transform(self, X):

        # Make a copy so that X is not changed

        X_copy = X.copy()

        

        count_encoded = np.log1p(self.count_encoder.transform(X[self.cat_cols]))

        X_copy = X_copy.join(count_encoded.add_suffix("_count"))

        

        target_encoded = self.target_encoder.transform(X[self.cat_cols])

        X_copy = X_copy.join(target_encoded.add_suffix("_target"))

        

        OH_X = pd.DataFrame(self.OH_encoder.transform(X[self.cat_cols]))

        OH_X.index = X_copy.index

        

        X_copy = X_copy.drop(self.cat_cols, axis=1)

        X_copy = pd.concat([X_copy, OH_X], axis=1)

        

        return X_copy

    

class Scaler(BaseEstimator, TransformerMixin):

    def __init__(self):

        return

    

    def fit(self, X, y=None):

        self.num_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]

        self.scaler = StandardScaler()

        for cname in self.num_cols:

            self.scaler.fit(X[[cname]]) 

        return self

    

    def transform(self, X):

        X_copy = X.copy()

        for cname in self.num_cols:

            X_copy[cname] = self.scaler.transform(X[[cname]])

        return X_copy
from sklearn.tree import DecisionTreeRegressor

from sklearn.feature_selection import SelectFromModel



class Select_Features(BaseEstimator, TransformerMixin):

    def __init__(self, threshold="1.1*mean"):

        self.threshold = threshold

    

    def fit(self, X, y=None):

        decision_tree = DecisionTreeRegressor(random_state=8)

        select_features = SelectFromModel(decision_tree, threshold=self.threshold) # Threshold parameter can be optimised in CV

        self.select_features = select_features.fit(X, y)

        return self

    

    def transform(self, X):

        X_copy = X.copy()

        X_copy = self.select_features.transform(X)

        feature_index = self.select_features.get_support()

        feature_name = X.columns[feature_index]

        return X_copy
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import GridSearchCV



# Set of parameters to adjust randomly for cross validation

params = {

    'model__n_estimators' : np.arange(100,2000,50),

    'model__learning_rate' : np.arange(0.01,0.2,0.01),

    'model__max_depth' : np.arange(1,4,1),

    'drop_features__drop_threshold' : np.arange(0.65,1.0, 0.05),

    'log_skew__skew_threshold' : np.arange(0.75, 1.0, 0.05)

}



model = XGBRegressor()



full_pipeline = Pipeline([

    ('drop_features', Drop_Features()),

    ('add_features', Add_Features()),

    ('log_skew', Skewed_Features()),

    ('remove_colinear', Colinear_Features()),

    ('impute', Imputer()),

    ('encode', Encoder()),

#     ('cat_interactions', Create_Interactions()),

    ('scale', Scaler()),

#     ('select_features', Select_Features()),

    ('model', model)

])



# Cross validation with a random search

param_search = RandomizedSearchCV(full_pipeline, param_distributions=params,

                                 n_iter=300,

                                 cv=5,

                                 scoring='neg_mean_squared_error',

                                 verbose=False,

                                 random_state=8)



param_search.fit(X_train, y)



# Print randomized CV results in a list

print("Random search:")

cv_results = param_search.cv_results_

for mean_score, params in zip(cv_results["mean_test_score"], cv_results["params"]):

    print(-1*mean_score, params)



print("\nBest random search results:")

print(-1*param_search.best_score_,param_search.best_params_)



# Best results:

# 0.014161684193906307 {'model__n_estimators': 500, 'model__max_depth': 2, 'model__learning_rate': 0.060000000000000005}



# Now use grid search to find more optimised hyper parameters

params = {

    'model__n_estimators' : np.arange(param_search.best_params_['model__n_estimators']-20,param_search.best_params_['model__n_estimators']+20,2),

    'model__learning_rate' : np.arange(param_search.best_params_['model__learning_rate']-0.01,param_search.best_params_['model__learning_rate']+0.01,0.005),

    'model__max_depth' : [param_search.best_params_['model__max_depth']],

    'drop_features__drop_threshold' : [param_search.best_params_['drop_features__drop_threshold']],

    'log_skew__skew_threshold' : [param_search.best_params_['log_skew__skew_threshold']]

}



# Best so far tuning

# 0.014075478857627461 {'model__learning_rate': 0.065, 'model__max_depth': 2, 'model__n_estimators': 493}



param_search = GridSearchCV(full_pipeline, param_grid=params,

                           cv=5,

                           scoring='neg_mean_squared_error',

                           verbose=False)



param_search.fit(X_train, y)



print("\nGrid search:")

cv_results = param_search.cv_results_

for mean_score, params in zip(cv_results["mean_test_score"], cv_results["params"]):

    print(-1*mean_score, params)

    

print("\nBest grid search results:")

print(-1*param_search.best_score_,param_search.best_params_)



# Refit the model using the whole dataset

param_search.refit



# Make predictions

preds = param_search.predict(X_test)

preds = np.expm1(preds)
# Save test predictions to file

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds})

output.to_csv('submission.csv', index=False)