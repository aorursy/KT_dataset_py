import pandas as pd, sklearn, numpy as np, os

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import SGDClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import VotingClassifier

from sklearn.metrics import accuracy_score, precision_score, make_scorer
data_folder = '../input'

train = pd.read_csv(os.path.join(data_folder, 'train.csv'))

test = pd.read_csv(os.path.join(data_folder, 'test.csv'))

n_train, m_train = train.shape
train.head()
train.info()
train.describe()
from sklearn.model_selection import train_test_split



def drop_unused_columns(df):

    return df.drop(['PassengerId', 'Cabin', 'Ticket', 'Embarked'], axis=1)



def to_features_and_labels(df):

    y = df['Survived'].values

    X = drop_unused_columns(df)

    X = X.drop('Survived', axis=1)

    return X, y



X_train_val, y_train_val = to_features_and_labels(train) # All data with labels, to be split into train and val

X_test = drop_unused_columns(test)



# Split the available training data into training set (used for choosing the best model) 

# and validation set (used for estimating the generalization error, could also be called "hold-out" set)

X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.20, random_state=42)

X_train.head()
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameColumnMapper(BaseEstimator, TransformerMixin):

    def __init__(self, column_name, mapping_func, new_column_name=None):

        self.column_name = column_name

        self.mapping_func = mapping_func

        self.new_column_name = new_column_name if new_column_name is not None else self.column_name

    def fit(self, X, y=None):

        # Nothing to do here

        return self

    def transform(self, X):

        transformed_column = X.transform({self.column_name: self.mapping_func})

        Y = X.copy()

        Y = Y.assign(**{self.new_column_name: transformed_column})

        if self.column_name != self.new_column_name:

            Y = Y.drop(self.column_name, axis=1)

        return Y



# Return a lambda function that extracts title from the full name, this allows instantiating the pattern only once

def extract_title():

    import re

    pattern = re.compile(', (\w*)')

    return lambda name: pattern.search(name).group(1)



# Example usage and output 

df = DataFrameColumnMapper(column_name='Name', mapping_func=extract_title(), new_column_name='Title').fit_transform(X_train)

df.head()
df['Title'].value_counts()[1:10]
class CategoricalTruncator(BaseEstimator, TransformerMixin):

    def __init__(self, column_name, n_values_to_keep=5):

        self.column_name = column_name

        self.n_values_to_keep = n_values_to_keep

        self.values = None

    def fit(self, X, y=None):

        # Here we must ensure that the test set is transformed similarly in the later phase and that the same values are kept

        self.values = list(X[self.column_name].value_counts()[:self.n_values_to_keep].keys())

        return self

    def transform(self, X):

        transform = lambda x: x if x in self.values else 'Other'

        y = X.transform({self.column_name: transform})

        return X.assign(**{self.column_name: y})



# Print title counts

title_counts = CategoricalTruncator('Title', n_values_to_keep=3).fit_transform(df)['Title'].value_counts()

title_counts
from sklearn.pipeline import Pipeline



pipeline = Pipeline([

    ('name_to_title', DataFrameColumnMapper(column_name='Name', mapping_func=extract_title(), new_column_name='Title')),

    ('truncate_titles', CategoricalTruncator('Title', n_values_to_keep=3))

])



df = pipeline.fit_transform(X_train)

df.head(10)
class ImputerByReference(BaseEstimator, TransformerMixin):

    def __init__(self, column_to_impute, column_ref):

        self.column_to_impute = column_to_impute

        self.column_ref = column_ref

        # TODO Allow specifying the aggregation function

        # self.impute_func = np.median if impute_type == 'median' or impute_type is None else np.mean

    def fit(self, X, y=None):

        # Pick columns of interest

        df = X.loc[:, [self.column_to_impute, self.column_ref]]

        # Dictionary containing mean per group

        self.value_per_group = df.groupby(self.column_ref).median().to_dict()[self.column_to_impute]

        return self

    def transform(self, X):

        def transform(row):

            row_copy = row.copy()

            if pd.isnull(row_copy.at[self.column_to_impute]):

                row_copy.at[self.column_to_impute] = self.value_per_group[row_copy.at[self.column_ref]]

            return row_copy

        return X.apply(transform, axis=1)



# Example output

ImputerByReference('Age', 'Title').fit_transform(df).head(10)
pipeline = Pipeline([

    ('name_to_title', DataFrameColumnMapper(column_name='Name', mapping_func=extract_title(), new_column_name='Title')),

    ('truncate_titles', CategoricalTruncator('Title', n_values_to_keep=3)),

    ('impute_ages_by_title', ImputerByReference('Age', 'Title'))

])



df = pipeline.fit_transform(X_train)

df.info()
class CategoricalToOneHotEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):

        self.columns = columns

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

    

        self.mappings = mappings

        return self



    def transform(self, X):

        Y = X.copy()

        for col in self.columns:

            transformed_col = Y.loc[:, col].transform(lambda x: self.mappings[col]['cat_to_int'][x])

            for key, val in self.mappings[col]['cat_to_int'].items():

                one_hot = (transformed_col == val) + 0 # Cast boolean to int by adding zero

                Y = Y.assign(**{'{}_{}'.format(col, key): one_hot})

            Y = Y.drop(col, axis=1)

        return Y

    

# Example output    

CategoricalToOneHotEncoder().fit_transform(df).head()   
from sklearn.preprocessing import MinMaxScaler

from sklearn.impute import SimpleImputer



class DataFrameToValuesTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):

        pass

    def fit(self, X, y=None):

        # Remember the order of attributes before converting to NumPy

        self.attribute_order = list(X)

        return self

    def transform(self, X):

        return X.loc[:, self.attribute_order].values



def build_preprocessing_pipeline():

    return Pipeline([

        ('name_to_title', DataFrameColumnMapper(column_name='Name', mapping_func=extract_title(), new_column_name='Title')),

        ('truncate_titles', CategoricalTruncator(column_name='Title', n_values_to_keep=3)),

        ('impute_ages_by_title', ImputerByReference(column_to_impute='Age', column_ref='Title')),

        ('encode_categorical_onehot', CategoricalToOneHotEncoder()),

        ('encode_pclass_onehot', CategoricalToOneHotEncoder(columns=['Pclass'])),

        ('to_numpy', DataFrameToValuesTransformer()),

        ('imputer', SimpleImputer(strategy='median')), # Test set has one missing fare

        ('scaler', MinMaxScaler())

    ])



X_train_prepared = build_preprocessing_pipeline().fit_transform(X_train)

print('Prepared training data: {} samples, {} features'.format(*X_train_prepared.shape))
def build_pipeline(classifier=None):

    preprocessing_pipeline = build_preprocessing_pipeline()

    return Pipeline([

        ('preprocessing', preprocessing_pipeline),

        ('classifier', classifier) # Expected to be filled by grid search

    ])





def build_grid_search(pipeline, param_grid):

    return GridSearchCV(pipeline, param_grid, cv=5, return_train_score=True, refit='accuracy',

                        scoring={ 'accuracy': make_scorer(accuracy_score),

                                  'precision': make_scorer(precision_score)

                                },

                        verbose=1)



def pretty_cv_results(cv_results, 

                      sort_by='rank_test_accuracy',

                      sort_ascending=True,

                      n_rows=5):

    df = pd.DataFrame(cv_results)

    cols_of_interest = [key for key in df.keys() if key.startswith('param_') 

                        or key.startswith('mean_train') 

                        or key.startswith('mean_test_')

                        or key.startswith('rank')]

    return df.loc[:, cols_of_interest].sort_values(by=sort_by, ascending=sort_ascending).head(n_rows)



def run_grid_search(grid_search):

    grid_search.fit(X_train, y_train)

    print('Best test score accuracy is:', grid_search.best_score_)

    return pretty_cv_results(grid_search.cv_results_)
param_grid = [

    { 'preprocessing__truncate_titles__n_values_to_keep': [3, 4, 5],

      'classifier': [SGDClassifier(loss='log', tol=None, random_state=42)],

      'classifier__alpha': np.logspace(-5, -3, 3),

      'classifier__penalty': ['l2'],

      'classifier__max_iter': [20],

    }

]

log_grid_search = build_grid_search(pipeline=build_pipeline(), param_grid=param_grid)

linear_cv = run_grid_search(grid_search=log_grid_search)
linear_cv
param_grid = [

    { 'preprocessing__truncate_titles__n_values_to_keep': [5],

      'classifier': [RandomForestClassifier(random_state=42)],

      'classifier__n_estimators': [10, 30, 100],

      'classifier__max_features': range(4, 14, 3)

    }

]

rf_grid_search = build_grid_search(pipeline=build_pipeline(), param_grid=param_grid)

rf_cv_results = run_grid_search(grid_search=rf_grid_search)

rf_cv_results
param_grid = [

    { 

        'preprocessing__truncate_titles__n_values_to_keep': [5],

        'classifier': [ SVC(random_state=42, probability=True) ], # Probability to use in voting later

        'classifier__C': np.logspace(-1, 1, 3),

        'classifier__kernel': ['linear', 'poly', 'rbf'],

        'classifier__gamma': ['auto', 'scale']

    }

]





svm_grid_search = build_grid_search(pipeline=build_pipeline(), param_grid=param_grid)

svm_cv_results = run_grid_search(grid_search=svm_grid_search)
svm_cv_results
from sklearn.gaussian_process.kernels import RBF, Matern



param_grid = [

    { 

        'preprocessing__truncate_titles__n_values_to_keep': [5],

        'classifier': [ GaussianProcessClassifier() ], 

        'classifier__kernel': [1.0*RBF(1.0), 1.0*Matern(1.0)]

    }

]



gp_grid_search = build_grid_search(pipeline=build_pipeline(), param_grid=param_grid)

gp_cv_results = run_grid_search(grid_search=gp_grid_search)

gp_cv_results
from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier



param_grid = [

    { 

        'preprocessing__truncate_titles__n_values_to_keep': [5],

        'classifier': [ AdaBoostClassifier(random_state=42) ],

        'classifier__n_estimators': [50, 100],

        'classifier__learning_rate': np.logspace(-1, 1, 3),

        'classifier__base_estimator': [

            DecisionTreeClassifier(max_depth=1),

            DecisionTreeClassifier(max_depth=2)

        ],

        # 'classifier__base_estimator__max_depth': [1, 2]

    }

]



ada_grid_search = build_grid_search(pipeline=build_pipeline(), param_grid=param_grid)

ada_cv_results = run_grid_search(grid_search=ada_grid_search)
ada_cv_results
param_grid = [

    { 

        'preprocessing__truncate_titles__n_values_to_keep': [5],

        'classifier': [ GradientBoostingClassifier(random_state=42) ],

        'classifier__loss': ['deviance'],

        'classifier__n_estimators': [50, 100],

        'classifier__max_features': [7, 13],

        'classifier__max_depth': [3, 5],

        'classifier__min_samples_leaf': [1],

        'classifier__min_samples_split': [2]

    }

]



gb_grid_search = build_grid_search(pipeline=build_pipeline(), param_grid=param_grid)

gb_cv_results = run_grid_search(grid_search=gb_grid_search)
gb_cv_results
gb_grid_search.best_estimator_.score(X_val, y_val)
voting_estimators = [

    # ('logistic', log_grid_search),

    # ('rf', rf_grid_search),

    ('svc', svm_grid_search),

    ('gp', gp_grid_search),

    # ('ada', ada_grid_search),

    ('gb', gb_grid_search),

]



estimators_with_names = [(name, grid_search.best_estimator_) for name, grid_search in voting_estimators]



voting_classifier = VotingClassifier(estimators=estimators_with_names,

                                     voting='soft')



voting_classifier.fit(X_train, y_train)

voting_classifier.score(X_val, y_val)

# cross_val_score(voting_classifier, X_train_val, y_train_val, cv=5)
voting_classifier.fit(X_train_val, y_train_val)
def get_predictions(estimator):

    predictions = estimator.predict(X_test)

    indices = test.loc[:, 'PassengerId']

    as_dict = [{'PassengerId': index, 'Survived': prediction} for index, prediction in zip(indices, predictions)]

    return pd.DataFrame.from_dict(as_dict)



predictions = get_predictions(voting_classifier)
submission_folder = '.'

dest_file = os.path.join(submission_folder, 'submission.csv')

predictions.to_csv(dest_file, index=False)