import numpy as np
import pandas as pd

import os

from typing import Tuple
# To ignore SettingWithCopyWarning
# When transforming a DataFrame in a Sci-kit Learn pipeline, this warning is thrown
pd.options.mode.chained_assignment = None
class DataHelper():
    """Simply read the input data into pd.DataFrames"""
    
    def __init__(self, directory: str ='../input/', train_file: str ='train.csv', test_file: str ='test.csv'):
        self.directory =directory
        self.test_file = test_file
        self.train_file = train_file
        pass
    
    def __read_csv(self, filename: str) -> pd.DataFrame:
        return pd.read_csv(self.directory + filename)
    
    def get_train_data(self, y_columns) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Reads the train data and returns a tuple containing X_train and y_train DataFrames splitted by y_columns."""
        train_data = self.__read_csv(self.train_file)
        return train_data.drop(y_columns, axis=1), train_data[y_columns]
    
    def get_test_data(self) -> pd.DataFrame:
        """Reads the test data into a DataFrame"""
        return self.__read_csv(self.test_file)
    
    def __format_prediction_before_save(self, index, prediction) -> pd.DataFrame:
        reshaped = np.array(prediction).reshape(418,1)
        return pd.DataFrame(reshaped, index=index, columns=['Survived'])
    
    def save_prediction(self, index, prediction: pd.DataFrame, filename: str = 'submission.csv'):
        self.__format_prediction_before_save(index, prediction).to_csv(path_or_buf=filename, sep=',', header=True)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
class CabinExtractor(BaseEstimator, TransformerMixin):
    """Map the Cabin column to 0 if the person hadn't a cabin and to 1 if he had one"""
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def __get_deck(self, cabin) -> int:
        return 0 if cabin == 0 else 1
    
    def transform(self, X, y=None):
        X.Cabin.fillna(0, inplace=True)
        X.Cabin = X.Cabin.apply(self.__get_deck)
        return X
class TitleExtractor(BaseEstimator, TransformerMixin):
    """Picks the title from the name and maps it to an general title using the title_mapping."""
    def __init__(self, regex: str):
        self.regex = regex
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X['Title'] = X.Name.str.extract(self.regex)[0].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        return X.drop('Name', axis=1)
class DataFrameSelector(BaseEstimator, TransformerMixin):
    """Selects the given columns"""
    def __init__(self, column_names):
        self.column_names = column_names
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X[self.column_names]
class DataFrameToValueConverter(BaseEstimator, TransformerMixin):
    """Converts a DataFrame into a numpy array"""
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.values
class FamilyAttributeCombiner(BaseEstimator, TransformerMixin):
    """Creates new column with 0 if they hadnt family members on board and 1 if they had"""
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def get_val(self, amount):
        if amount > 0:
            return 1
        else:
            return 0
    
    def transform(self, X, y=None):
        X['HasFamily'] = X[['SibSp','Parch']].sum(axis=1).apply(self.get_val)
        return X.drop(['SibSp','Parch'], axis=1)
class GridSearchHelper():
    def __init__(self, preprocesing_pipeline: Pipeline, X: pd.DataFrame, y: pd.DataFrame):
        self.preprocesing_pipeline = preprocesing_pipeline
        self.X = X
        self.y = y
        pass
    
    def __get_grid_search(self, clf, param_grid: list, kwargs) -> GridSearchCV:
        return GridSearchCV(clf, param_grid, **kwargs)
    
    def __get_best_estimator(self, clf, param_grid: list, kwargs) -> BaseEstimator:
        grid_search = self.__get_grid_search(clf, param_grid, kwargs)
        grid_search.fit(self.preprocesing_pipeline.fit_transform(self.X), self.y)
        return grid_search.best_estimator_

    def get_best_classifier(self, clf, param_grid, **kwargs):
        return self.__get_best_estimator(clf, param_grid, kwargs)
not_needed_attribs = ['PassengerId', 'Name', 'Ticket','Cabin', 'Embarked']
cat_attribs = ['Sex', 'Pclass'] #['Sex', 'Pclass', 'Cabin','Name', 'SibSp', 'Parch',]
num_attribs = ['SibSp', 'Parch', 'Age', 'Fare'] #['Age', 'Fare',]
# from https://www.kaggle.com/acsrikar279/titanic-higher-score-using-kneighborsclassifier/notebook
title_regex = '([A-Za-z]+)\.'
num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('df_to_value_converter',DataFrameToValueConverter()),
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('std_scaler', StandardScaler()),
])
cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
#    ('family_attribute_combiner',FamilyAttributeCombiner()),
#    ('title_extractor', TitleExtractor(title_regex)),
#    ('cabin_extractor', CabinExtractor()),
    ('df_to_value_converter',DataFrameToValueConverter()),
    ('one_hot_encoder', OneHotEncoder()),
])
feature_union = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline),
])
preprocessing_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs + cat_attribs)),
    ('feature_union', feature_union),
])
dataHelper = DataHelper()
X_train, y_train = dataHelper.get_train_data(['Survived'])
X_test = dataHelper.get_test_data()
preprocessing_pipeline.fit(X_train)
gridSearchHelper = GridSearchHelper(preprocessing_pipeline, X_train, y_train.values.ravel())
def get_estimator_and_print_score(clf, param_grid):
    X_prepared = preprocessing_pipeline.transform(X_train)
    estimator = gridSearchHelper.get_best_classifier(clf, param_grid, cv=2, scoring='accuracy', refit=True)
    estimator.fit(X_prepared, y_train.values.ravel())
    scores = cross_val_score(estimator, X_prepared, y_train.values.ravel(), cv=2, scoring='accuracy')
    print(type(clf).__name__)
    print('Crossval:', scores.mean())
    print('Accuracy:',accuracy_score(y_train, estimator.predict(X_prepared)))
    print('F1-score:', f1_score(y_train, estimator.predict(X_prepared)))
    return estimator
svc_param_grid = [
    {
        'C':[0.9,1.0,1.1],
        'gamma':[1]
    },
]
rfc_param_grid = [
    # {
        # 'max_features': [None, 'auto', 'sqrt', 'log2'],
        # 'max_depth': [None,4,5,6,7,8],
        # 'min_samples_split':[2],
        # 'oob_score': [True],
        # 'criterion' :['gini', 'entropy'],
        # 'random_state':[42],
        # 'n_estimators':[100],
    # },
    {
        'max_depth' : [10], 'max_features' : ['log2'], 'max_leaf_nodes' : [None],
            'min_impurity_decrease' : [0.0], 'min_impurity_split' : [None],
            'min_samples_leaf' : [1], 'min_samples_split' : [30],
            'min_weight_fraction_leaf' : [0.0], 'n_estimators' : [100], 'n_jobs' : [1],
            'oob_score' : [False], 'random_state' : [None], 'verbose' : [0], 'warm_start' : [False]
    }
]
dtc_param_grid = [
    {
        'max_depth': [4, 5],
    }
]
sgdc_param_grid = [
    {
        'shuffle': [True],
        'max_iter':[np.ceil(10**6 / len(X_train)),],
    },
]
knc_param_grid = [
    {
        'n_neighbors':[2,3,4,5,7],
        'weights':['uniform','distance'],
        # 'leaf_size':[20,30,40]
    },
    {
        
    }
]
# current_estimator = get_estimator_and_print_score(SVC(), svc_param_grid)
# current_estimator = get_estimator_and_print_score(RandomForestClassifier(), rfc_param_grid)
# current_estimator = get_estimator_and_print_score(DecisionTreeClassifier(), dtc_param_grid)
# current_estimator = get_estimator_and_print_score(SGDClassifier(), sgdc_param_grid)
current_estimator = get_estimator_and_print_score(KNeighborsClassifier(), knc_param_grid)
final_pipeline = Pipeline([
    ('preprocessing', preprocessing_pipeline),
    ('estimator', current_estimator)
])
final_pipeline.fit(X_train, y_train.values.ravel())
dataHelper.save_prediction(X_test['PassengerId'], final_pipeline.predict(X_test))