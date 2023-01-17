import os



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb



from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier

from sklearn.metrics import make_scorer, cohen_kappa_score, accuracy_score

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score



from xgboost import XGBClassifier



%matplotlib inline

plt.rc('figure', figsize=(20.0, 10.0))
INPUT_DIR = "../input"

print(os.listdir(INPUT_DIR))
train_df = pd.read_csv(os.path.join(INPUT_DIR, 'train', 'train.csv'))

X_test = pd.read_csv(os.path.join(INPUT_DIR, 'test', 'test.csv'))
train_df.hist(figsize=(20, 10))

plt.tight_layout()
class DataFrameColumnMapper(BaseEstimator, TransformerMixin):

    """

    Map DataFrame column to a new column (similar to DataFrameMapper from sklearn-pandas)

    

    Attributes:

        column_name (str): Column name to transform

        mapping_func (func): Function to apply to given column values

        new_column_name (str): Name for the new column, leave empty if replacing `column_name`

        drop_original (bool): Drop original column if true and new_column_name != column_name

    """

    def __init__(self, column_name, mapping_func, new_column_name=None, drop_original=True):

        """

        """

        self.column_name = column_name

        self.mapping_func = mapping_func

        self.new_column_name = new_column_name if new_column_name is not None else self.column_name

        self.drop_original = drop_original



    def fit(self, X, y=None):

        return self



    def transform(self, X):

        transformed_column = X.transform({self.column_name: self.mapping_func})

        Y = X.copy()

        Y = Y.assign(**{self.new_column_name: transformed_column})

        if self.column_name != self.new_column_name and self.drop_original:

            Y = Y.drop(self.column_name, axis=1)

        return Y
class CategoricalToOneHotEncoder(BaseEstimator, TransformerMixin):

    """

    One-hot encode given columns.

    

    Attributes:

        columns (List[str]): Columns to one-hot encode.

        mappings_ (Dict[str, Dict]): Mapping from original column name to the one-hot-encoded column names

    """

    def __init__(self, columns=None):

        self.columns = columns

        self.mappings_ = None

    def fit(self, X, y=None):

        # Pick all categorical attributes if no columns to transform were specified

        if self.columns is None:

            self.columns = X.select_dtypes(exclude='number')

        

        # Keep track of which categorical attributes are assigned to which integer. This is important 

        # when transforming the test set.

        mappings = {}

        

        for col in self.columns:

            labels, uniques = X.loc[:, col].factorize() # Assigns unique integers for all categories

            int_and_cat = list(enumerate(uniques))

            cat_and_int = [(x[1], x[0]) for x in int_and_cat]

            mappings[col] = {'int_to_cat': dict(int_and_cat), 'cat_to_int': dict(cat_and_int)}

    

        self.mappings_ = mappings

        return self



    def transform(self, X):

        Y = X.copy()

        for col in self.columns:

            transformed_col = Y.loc[:, col].transform(lambda x: self.mappings_[col]['cat_to_int'][x])

            for key, val in self.mappings_[col]['cat_to_int'].items():

                one_hot = (transformed_col == val) + 0 # Cast boolean to int by adding zero

                Y = Y.assign(**{'{}_{}'.format(col, key): one_hot})

            Y = Y.drop(col, axis=1)

        return Y
class CategoricalTruncator(BaseEstimator, TransformerMixin):

    """

    Keep only N most frequent categories for a given column, replace others with "Other"

    

    Attributes:

        column_name (str): Column for which to truncate categories

        n_values_to_keep (int): How many of the most frequent values to keep (1 for keeping only most frequent, etc.)

        values_ (List[str]): List of category names to keep, others are replaced with "Other"

    """

    def __init__(self, column_name, n_values_to_keep=5):

        self.column_name = column_name

        self.n_values_to_keep = n_values_to_keep

        self.values_ = None

    def fit(self, X, y=None):

        # Here we must ensure that the test set is transformed similarly in the later phase and that the same values are kept

        self.values_ = list(X[self.column_name].value_counts()[:self.n_values_to_keep].keys())

        return self

    def transform(self, X):

        transform = lambda x: x if x in self.values_ else 'Other'

        Y = X.copy()

        y = Y.transform({self.column_name: transform})

        return Y.assign(**{self.column_name: y})
class DataFrameColumnDropper(BaseEstimator, TransformerMixin):

    """

    Drop given columns.

    

    Attributes:

        column_names (List[Str]): List of columns to drop

    """

    def __init__(self, column_names):

        self.column_names = column_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X.copy().drop(self.column_names, axis=1)
class ColumnByFeatureImportancePicker(BaseEstimator, TransformerMixin):

    """

    Pick columns by feature importance

    Attributes:

        n_features (Optional[int]): How many most important features to keep, None for noop transformation

        classifier (: Classifier, must have `feature_importances_` available after `fit` has been called

    """

    def __init__(self, n_features: int = 20, classifier=RandomForestClassifier(n_estimators=100, random_state=42)):

        self.n_features = n_features

        self.classifier = classifier

        self.attributes_ = None

        

    def fit_and_compute_importances(self, X_df, y):

        """

        :return: Sorted list of tuples containing column name and its feature importance

        """

        X_numeric = X_df.select_dtypes(include='number')

        X = X_numeric.values

        self.classifier.fit(X, y)

        feature_importances = self.classifier.feature_importances_

        feature_names = list(X_numeric)

        feature_importances_with_names = [(feature_name, feature_importance) for feature_name, feature_importance in zip(feature_names, feature_importances)]

        feature_importances_with_names.sort(key=lambda x: x[1], reverse=True)

        return feature_importances_with_names

        

    def fit(self, X, y=None):

        if self.n_features is None:

            # Do nothing but keep the order

            self.attributes_ = list(X)

            return self

        

        assert y is not None, "Feature importances cannot be computed without y!"

        feature_importances_with_names = self.fit_and_compute_importances(X, y)

        self.attributes_ = [feature_name for feature_name, _ in feature_importances_with_names[:self.n_features]]

        return self

    

    def transform(self, X):

        return X.copy().loc[:, self.attributes_]
class DataFrameToValuesTransformer(BaseEstimator, TransformerMixin):

    """

    Transform DataFrame to NumPy array.

    

    Attributes:

        attributes_ (List[str]): List of DataFrame column names

    """

    def __init__(self):

        self.attributes_ = None

        pass

    def fit(self, X, y=None):

        # Remember the order of attributes before converting to NumPy to ensure the columns

        # are included in the same order when transforming validation or test dataset

        self.attributes_ = list(X)

        return self

    def transform(self, X):

        return X.loc[:, self.attributes_].values
from sklearn.model_selection import train_test_split



def to_features_and_labels(df):

    y = df['AdoptionSpeed'].values

    X = df.drop('AdoptionSpeed', axis=1)

    return X, y



X_train_val, y_train_val = to_features_and_labels(train_df)



X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.20, random_state=42,

                                                  stratify=y_train_val)



print("Shape of X_train:", X_train.shape)

print("Shape of X_val:", X_val.shape)

print("Shape of y_train:", y_train.shape)

print("Shape of y_val:", y_val.shape)
X_train.head()
def has_field_transformer(column_name, new_column_name=None, is_missing_func=pd.notna) -> TransformerMixin:

    return DataFrameColumnMapper(column_name=column_name,

                                 mapping_func=lambda name: np.int(is_missing_func(name)),

                                 drop_original=True,

                                 new_column_name=new_column_name if new_column_name is not None else column_name)



def value_matches_transformer(column_name, new_column_name=None, matches=pd.notna) -> TransformerMixin:

    return DataFrameColumnMapper(column_name=column_name,

                                 mapping_func=lambda value: np.int(matches(value)),

                                 drop_original=False,

                                 new_column_name=new_column_name if new_column_name is not None else column_name)



def map_categories(column_name, mapping_dict) -> TransformerMixin:

    return DataFrameColumnMapper(column_name=column_name,

                                 mapping_func=lambda x: mapping_dict[x])



def onehot_encode(columns) -> TransformerMixin:

    return CategoricalToOneHotEncoder(columns=columns)



def truncate_categorical(column_name, n_values_to_keep=10):

    return CategoricalTruncator(column_name=column_name, n_values_to_keep=n_values_to_keep)



ONEHOT_ENCODED_COLUMNS = ["Type", "Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3", "Health",

                          "FurLength", "Vaccinated", "Dewormed", "Sterilized", "State", "RescuerID"]



def build_preprocessing_pipeline() -> Pipeline:

     return Pipeline([

        ('add_has_name', has_field_transformer(column_name="Name", new_column_name="hasName")),

        ('add_is_free', value_matches_transformer(column_name="Fee", new_column_name="isFree",

                                                  matches=lambda value: value < 1)),

        ('map_type_to_species', map_categories(column_name="Type", mapping_dict={1: 'dog', 2: 'cat'})),

        ('map_gender_to_names', map_categories(column_name="Gender", mapping_dict={1: 'male', 2: 'female', 3: 'mixed'})),

        ('truncate_breed1', truncate_categorical(column_name="Breed1", n_values_to_keep=10)),

        ('truncate_breed2', truncate_categorical(column_name="Breed2", n_values_to_keep=10)),

        ('truncate_state', truncate_categorical(column_name="State", n_values_to_keep=10)),

        ('truncate_rescuer_id', truncate_categorical(column_name="RescuerID", n_values_to_keep=10)),

        ('onehot_encode', CategoricalToOneHotEncoder(columns=ONEHOT_ENCODED_COLUMNS)),

        ('drop_unused_columns', DataFrameColumnDropper(

            column_names=['PetID', 'Description', 'Type_dog'

        ])),

        ('pick_columns_by_importance', ColumnByFeatureImportancePicker(n_features=None))

    ])



preprocessing_pipeline = build_preprocessing_pipeline()

X_train_preprocessed = preprocessing_pipeline.fit_transform(X_train, y_train)

X_val_preprocessed = preprocessing_pipeline.transform(X_val)



X_train_preprocessed.head(10)
print("Number of features:", len(list(X_train_preprocessed)))

print("")



print("Numerical columns:", list(X_train_preprocessed.select_dtypes(include="number")))

print("")



print("Non-numerical columns:", list(X_train_preprocessed.select_dtypes(exclude="number")))
X_train_preprocessed.info()
def build_preparation_pipeline():

    return Pipeline([

        ('to_numpy', DataFrameToValuesTransformer()),

        ('scaler', StandardScaler())

    ])



def build_full_pipeline(classifier=None):

    preprocessing_pipeline = build_preprocessing_pipeline()

    preparation_pipeline = build_preparation_pipeline()

    return Pipeline([

        ('preprocessing', preprocessing_pipeline),

        ('preparation', preparation_pipeline),

        ('classifier', classifier)  # Expected to be filled by parameter search

    ])
def compute_feature_importances(classifier):

    """

    :param classifier: Classifier to use for computing feature importances, must have `feature_importances_` attribute

    :return: List of tuples containing column name and its feature importance

    """

    pipeline = build_full_pipeline(classifier=classifier)

    pipeline.fit(X_train, y_train)

    assert hasattr(classifier, 'feature_importances_')

    feature_importances = classifier.feature_importances_

    feature_names = pipeline.named_steps['preparation'].named_steps['to_numpy'].attributes_

    feature_importances_with_names = [(feature_name, feature_importance) for feature_name, feature_importance in zip(feature_names, feature_importances)]

    feature_importances_with_names.sort(key=lambda x: x[1], reverse=True)

    return feature_importances_with_names



rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=None)

feature_importances_with_names = compute_feature_importances(rf_classifier)



N_MOST_IMPORTANT_TO_SHOW = 50

print("Feature importances (top {}):".format(N_MOST_IMPORTANT_TO_SHOW))

for feature_name, feature_importance in feature_importances_with_names[:N_MOST_IMPORTANT_TO_SHOW]:

    print("{} -> {}".format(feature_name, feature_importance))

    

rf_pipeline = build_full_pipeline(classifier=rf_classifier)

cross_val_score(rf_pipeline, X_train, y_train, cv=5, scoring=make_scorer(cohen_kappa_score))
rf_classifier = RandomForestClassifier(n_estimators=100)

rf_pipeline = build_full_pipeline(classifier=rf_classifier)

cross_val_score(rf_pipeline, X_train, y_train, cv=5, scoring=make_scorer(cohen_kappa_score))
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix



# From https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    import itertools

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    # print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.tight_layout()

    

    

y_pred = cross_val_predict(rf_pipeline, X=X_train, y=y_train, cv=5)



cnf_matrix = confusion_matrix(y_true=y_train, y_pred=y_pred)



plt.figure(figsize=(15, 7))

plot_confusion_matrix(cnf_matrix, classes=range(0, 5),

                      title='Confusion matrix, without normalization')
def build_search(pipeline, param_distributions, n_iter=10):

    return RandomizedSearchCV(pipeline, param_distributions=param_distributions, 

                              cv=5, return_train_score=True, refit='cohen_kappa',

                              n_iter=n_iter,

                              scoring={

                                    'accuracy': make_scorer(accuracy_score),

                                    'cohen_kappa': make_scorer(cohen_kappa_score)

                               },

                              verbose=1, random_state=42)



def pretty_cv_results(cv_results, 

                      sort_by='rank_test_cohen_kappa',

                      sort_ascending=True,

                      n_rows=20):

    df = pd.DataFrame(cv_results)

    cols_of_interest = [key for key in df.keys() if key.startswith('param_') 

                        or key.startswith('mean_train') 

                        or key.startswith('mean_test_')

                        or key.startswith('rank')]

    return df.loc[:, cols_of_interest].sort_values(by=sort_by, ascending=sort_ascending).head(n_rows)



def run_search(search):

    search.fit(X_train, y_train)

    print('Best score is:', search.best_score_)

    return pretty_cv_results(search.cv_results_)
param_distributions = {

        'preprocessing__pick_columns_by_importance__n_features': [None, 50],

        'classifier': [RandomForestClassifier(n_estimators=250, random_state=42, max_depth=10)],

        'classifier__max_depth': [10, None]

    }



rf_feature_search = build_search(build_full_pipeline(), param_distributions=param_distributions, n_iter=4)

rf_feature_cv_results = run_search(search=rf_feature_search)

rf_feature_cv_results
param_distributions = {

        'preprocessing__pick_columns_by_importance__n_features': [None],

        'classifier': [RandomForestClassifier(n_estimators=500, random_state=42)],

        'classifier__n_estimators': [500],

        'classifier__max_features': ['auto', 'log2'],

        'classifier__max_depth': [None, 10],

        'classifier__bootstrap': [False, True],

        'classifier__min_samples_leaf': [1, 5, 10],

        'classifier__min_samples_split': [2, 5, 10],

        'classifier__criterion': ['gini', 'entropy'],

    }



rf_search = build_search(build_full_pipeline(), param_distributions=param_distributions, n_iter=50)

rf_cv_results = run_search(search=rf_search)

rf_cv_results
param_distributions = {

        'preprocessing__pick_columns_by_importance__n_features': [None],

        'classifier': [ExtraTreesClassifier(n_estimators=500, random_state=42)],

        'classifier__n_estimators': [500],

        'classifier__max_features': ['auto', 'log2'],

        'classifier__max_depth': [None],

        # 'classifier__bootstrap': [False, True],

        # 'classifier__min_samples_leaf': [0.001, 0.01, 0.05, 1, 5, 10],

        'classifier__min_samples_split': [2, 5, 10],

        'classifier__criterion': ['gini', 'entropy'],

    }



et_search = build_search(build_full_pipeline(), param_distributions=param_distributions, n_iter=50)

et_cv_results = run_search(search=et_search)

et_cv_results
from sklearn.linear_model import LogisticRegression



param_distributions = {

        'preprocessing__pick_columns_by_importance__n_features': [None, 50],

        'classifier': [LogisticRegression(solver='lbfgs', random_state=42)],

        'classifier__multi_class': ['ovr', 'multinomial'],

        'classifier__C': np.logspace(-3, 0, 4),

    }



logistic_search = build_search(build_full_pipeline(), param_distributions=param_distributions, n_iter=20)

logistic_cv_results = run_search(search=logistic_search)

logistic_cv_results
from sklearn.neural_network import MLPClassifier



param_distributions = {

        'preprocessing__pick_columns_by_importance__n_features': [None, 50],

        'preparation__scaler': [MinMaxScaler()],

        'classifier': [MLPClassifier(hidden_layer_sizes=(100, ), random_state=42)],

        'classifier__hidden_layer_sizes': [[10], [10, 10,], [10, 10, 10]],

        'classifier__alpha': np.logspace(-4, -2, 3),

        'classifier__solver': ['adam'],

        'classifier__tol': np.logspace(-4, -2, 3),

        'classifier__learning_rate_init': np.logspace(-3, -1, 3),

        'classifier__activation': ['relu', 'tanh'],

    }



mlp_search = build_search(build_full_pipeline(), param_distributions=param_distributions, n_iter=5)

mlp_cv_results = run_search(search=mlp_search)

mlp_cv_results
"""

from sklearn.svm import SVC

param_distributions = { 

        'preprocessing__pick_columns_by_importance__n_features': [None],

        'classifier': [ SVC(random_state=42, probability=True) ], # Probability to use soft voting later

        'classifier__C': np.logspace(-1, 1, 3),

        'classifier__kernel': ['linear', 'poly', 'rbf'],

        'classifier__gamma': ['auto', 'scale']

    }





svm_search = build_search(pipeline=build_full_pipeline(), param_distributions=param_distributions, n_iter=1)

svm_cv_results = run_search(search=svm_search)

svm_cv_results

"""
from sklearn.ensemble import GradientBoostingClassifier



param_distributions = { 

        'preprocessing__pick_columns_by_importance__n_features': [None],

        'classifier': [ GradientBoostingClassifier(random_state=42) ],

        'classifier__loss': ['deviance'],

        'classifier__n_estimators': [100, 300],

        'classifier__max_features': ['log2', None],

        'classifier__max_depth': [5, 10],

        'classifier__min_samples_leaf': [1, 5, 10],

        'classifier__min_samples_split': [2, 5, 10],

        'classifier__learning_rate': [0.1, 0.2],

        'classifier__subsample': [0.75, 0.90, 1.0]

    }



gb_search = build_search(pipeline=build_full_pipeline(), param_distributions=param_distributions, n_iter=20)

gb_cv_results = run_search(search=gb_search)

gb_cv_results
param_distributions = {

    'preprocessing__pick_columns_by_importance__n_features': [None],

    'classifier': [ lgb.sklearn. LGBMClassifier(random_state=42, objective='multiclass') ],

    'classifier__boosting_type': ['gbdt', 'dart'],

    'classifier__num_leaves': [20, 31, 50],

    'classifier__max_depth': [-1],

    'classifier__learning_rate': [0.1, 0.2],

    'classifier__n_estimators': [100, 300],

    'classifier__subsample': [1.0, 0.9],

    'classifier__reg_alpha': [0.0, *np.logspace(-3, -2, 2)],

    'classifier__reg_lambda': [0.0],

}



# cross_val_score(lgbm_pipeline, X_train, y_train, cv=5, scoring=make_scorer(cohen_kappa_score))

# compute_feature_importances(classifier=lgbm_classifier)



lgbm_search = build_search(pipeline=build_full_pipeline(), param_distributions=param_distributions, n_iter=20)

lgbm_cv_results = run_search(search=lgbm_search)

lgbm_cv_results
param_distributions = {

    'preprocessing__pick_columns_by_importance__n_features': [None],

    'classifier': [ XGBClassifier(random_state=42) ],

    'classifier__max_depth': [3, 5],

    'classifier__learning_rate': [0.1, 0.2],

    'classifier__n_estimators': [100, 200],

    'classifier__reg_alpha': [0, 1e-3],

    'classifier__lambda': [1],

}



# cross_val_score(lgbm_pipeline, X_train, y_train, cv=5, scoring=make_scorer(cohen_kappa_score))

# compute_feature_importances(classifier=lgbm_classifier)



xgb_search = build_search(pipeline=build_full_pipeline(), param_distributions=param_distributions, n_iter=20)

xgb_cv_results = run_search(search=xgb_search)

xgb_cv_results
def cross_val_predictions(classifiers, X, y):

    """

    Stack all cross validation prediction probabilities from classifiers into a single matrix.

    Predictions are computed using `cross_val_predict`, ensuring that predictions are clean.

    """

    return np.hstack([cross_val_predict(classifier, X, y, cv=5, method='predict_proba') for classifier in classifiers])

 

def first_level_predictions(classifiers, X):

    """

    Stack all prediction probabilities from classifier probability predictions.

    """

    return np.hstack([classifier.predict_proba(X) for classifier in classifiers])



best_estimators = [

    rf_search.best_estimator_,

    et_search.best_estimator_,

    gb_search.best_estimator_,

    lgbm_search.best_estimator_,

    xgb_search.best_estimator_

]



X_train_second_level = cross_val_predictions(classifiers=best_estimators, X=X_train, y=y_train)



stacking_classifier = GradientBoostingClassifier(random_state=42)



stacking_classifier.fit(X_train_second_level, y_train)
for estimator in best_estimators:

    estimator.fit(X_train, y_train)



X_val_second_level = first_level_predictions(best_estimators, X_val)

y_val_pred = stacking_classifier.predict(X_val_second_level)



print("Performance of stacking classifier on the hold-out set:", cohen_kappa_score(y_val, y_val_pred))
X_train_val_second_level = cross_val_predictions(best_estimators, X=X_train_val, y=y_train_val)

stacking_classifier.fit(X_train_val_second_level, y_train_val)



for estimator in best_estimators:

    estimator.fit(X_train_val, y_train_val)



X_test_second_level = first_level_predictions(best_estimators, X=X_test)
def get_predictions(estimator, X):

    predictions = estimator.predict(X)

    indices = X_test.loc[:, 'PetID']

    as_dict = [{'PetID': index, 'AdoptionSpeed': prediction} for index, prediction in zip(indices, predictions)]

    df = pd.DataFrame.from_dict(as_dict)

    df = df.reindex(['PetID', 'AdoptionSpeed'], axis=1)

    return df



predictions = get_predictions(stacking_classifier, X=X_test_second_level)
def write_submission(predictions):

    submission_folder = '.'

    dest_file = os.path.join(submission_folder, 'submission.csv')

    predictions.to_csv(dest_file, index=False)

    print("Wrote to {}".format(dest_file))

    

write_submission(predictions)