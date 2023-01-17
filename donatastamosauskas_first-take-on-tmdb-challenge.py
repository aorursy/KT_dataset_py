import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# print(os.listdir('../input/'))
# print(os.listdir('./'))

training_data_set = pd.read_csv('../input/train.csv')
testing_data_set = pd.read_csv('../input/test.csv')

print("Training data set size is {} samples.".format(len(training_data_set)))
print("Test data set size is {} samples.".format(len(testing_data_set)))
training_data_set.head()
interesting_datapoints = pd.DataFrame(training_data_set[training_data_set.budget == training_data_set.budget.max()])
interesting_datapoints = interesting_datapoints.append(training_data_set[training_data_set.budget == training_data_set.budget.min()][0:1])
interesting_datapoints = interesting_datapoints.append(training_data_set[training_data_set.revenue == training_data_set.revenue.max()])
interesting_datapoints = interesting_datapoints.append(training_data_set[training_data_set.revenue == training_data_set.revenue.min()][0:1])
interesting_datapoints = interesting_datapoints.append(training_data_set[training_data_set.revenue / training_data_set.budget == max(training_data_set.revenue / [budget_row if budget_row > 1000 else -1 for budget_row in training_data_set.budget])])

interesting_datapoints
subplot, plots = plt.subplots(3, 2)
subplot.set_figheight(50)
subplot.set_figwidth(44)

plots[0][0].plot(training_data_set.revenue.sort_values().reset_index().revenue)
plots[0][0].set_title('Revenue', fontsize=24)

plots[0][1].plot(np.sqrt(np.sqrt(training_data_set.revenue.sort_values().reset_index().revenue)))
plots[0][1].set_title('Revenue sqrt of sqrt', fontsize=24)

plots[1][0].plot(training_data_set.budget.sort_values().reset_index().budget)
plots[1][0].set_title('Budget', fontsize=24)

plots[1][1].plot(np.log(training_data_set.budget.sort_values().reset_index().budget))
plots[1][1].set_title('Budget Log', fontsize=24)

plots[2][0].plot(training_data_set.popularity.sort_values().reset_index().popularity)
plots[2][0].set_title('Popularity', fontsize=24)

plots[2][1].plot(np.sqrt(np.sqrt(training_data_set.popularity.sort_values().reset_index().popularity)))
plots[2][1].set_title('Popularity sqrt of sqrt', fontsize=24)

plt.show()
print("Numerical columns: {}\n".format(training_data_set.select_dtypes(include=[np.number]).columns.tolist()))
print("Which columns have missing data: \n{}".format(training_data_set.isna().any()))
training_data_set['original_language'].unique()
def get_release_month(data):
    release_months = [
        # Through out the whole data set (train + test) there's only one record
        # without release_date, so hardcoding the most common value doesn't seem
        # that outrageous 
        int(date.split('/')[0]) if date is not None else 12 
        for date in data['release_date']
    ]
    return release_months

def append_release_month_col(data, release_month):
    modified_data = data.copy()
    modified_data['release_month'] = release_month
    return modified_data

print("Trainig set rows without date: {}".format(training_data_set['release_date'].isna().any().sum()))
print("Testing set rows without date: {}".format(testing_data_set['release_date'].isna().any().sum()))

release_months = get_release_month(training_data_set)

plt.hist(release_months)
plt.show()


def get_dircetor(data):
    import ast
    director_list = []
    
    if data.columns.contains('crew'):
        crew_data = data['crew'].copy()
    else:
        crew_data = data.copy()
        
    for crew_info in crew_data:
        if type(crew_info) == type(''):
            directors = [
                [employee['id'], employee['name']]
                 for employee in ast.literal_eval(crew_info) 
                 if employee['job'] == 'Director']
            
            if not directors:
                director_list.append([[-1, 'None']])
            else:
                director_list.append(directors)
                
        else:
            director_list.append([[-1, 'None']])
        
    return director_list 

def append_director_col(data, director_list):
    modified_data = data.copy()
    modified_data['director'] = director_list
    return modified_data


unique_directors_train = pd.Series((director[0][0] for director in get_dircetor(training_data_set))).unique().size
unique_directors_test = pd.Series((director[0][0] for director in get_dircetor(testing_data_set))).unique().size

print("Movies in training set: {}, unique directors: {}. One director on average directs {:.2f} movies."
      .format(len(training_data_set), unique_directors_train, len(training_data_set) / unique_directors_train))
print("Movies in testing set: {}, unique directors: {}. One director on average directs {:.2f} movies."
      .format(len(testing_data_set), unique_directors_test, len(testing_data_set) / unique_directors_test))
from sklearn.model_selection import train_test_split

def separate_only_numeric(data):
    """ Returns DataFrame with only numeric columns
    
    Arguments:
    data - DataFrame that will be used as source
    """
    return data.select_dtypes(include=[np.number])

def exclude_na_rows(data):
    """ Returns DataFrame where rows with na values have been removed
    
    Arguments:
    data - DataFrame that will be used as source
    """
    return data.dropna(axis='index')

def extract_label(data, columns=None, separate_label=True, label_name='revenue'):
    """ Separates the label from the data if specified, and selects only specified columns
    
    Arguments:
    data - array of data to be processed
    columns - columns to be left after preprocessing
    separate_label - whether to separate the label from the data and return it as a second parameter
    label_name - the name of the label column
    
    Returns: data_X, data_y
    """
    data_copy = data.copy()
    if separate_label: 
        data_y = data_copy.pop(label_name)
    else:
        data_y = [0 for _ in range(len(data_copy))]
    
    if columns is None:
        columns = data_copy.columns.difference(['id']).tolist()
    data_X = data_copy[columns]
    return data_X, data_y

def transform_dataset(data, test_size=0.3, only_numeric=True, delete_nan=True, separate_label=True):
    """ Preprocesses the data
    
    Arguments:
    data - array of data to be processed
    train_size - percentage (0 to 1) of data to be used as training data
    only_numeric - whether to use only numeric data
    training_data - whether this data has labels
    
    Returns: train_X, test_X, train_y, test_y
    """
    if only_numeric: 
        data = separate_only_numeric(data)
    if delete_nan:
        data = exclude_na_rows(data)
    data_X, data_y  = extract_label(data, separate_label=separate_label)
    return train_test_split(data_X, data_y, test_size=test_size, random_state=1)
from sklearn.base import TransformerMixin, BaseEstimator

class NumericDataSelector(TransformerMixin, BaseEstimator):
        
    def fit(self, data, labels=None):
        return self
        
    def transform(self, data, labels=None):
        return separate_only_numeric(data)
    
class FeatureSelector(TransformerMixin, BaseEstimator):
    
    def __init__(self, feature_names=None, remove_feature_names=None):
        self._feature_names = feature_names 
        self._remove_feature_names = remove_feature_names 
        
    def fit(self, data, labels=None):
        return self
        
    def transform(self, data, labels=None):
        if self._feature_names is None:
            self._feature_names = data.columns.tolist()
        
        if self._remove_feature_names is not None:
            self._feature_names = [feat for feat in self._feature_names if feat not in self._remove_feature_names]
        
        return data[self._feature_names]
    
class RootScaler(TransformerMixin, BaseEstimator):
    
    def __init__(self, scale_features=None, root_pow=4):
        self._scale_features = scale_features
        self._root_pow = root_pow
        
    def fit(self, data, labels=None):
        return self
    
    def transform(self, data, labels=None):
        if self._scale_features is None:
            self._scale_features = data.columns.tolist()
        
        scaled_data = data[self._scale_features]
        scaled_labels = labels
        
        for _ in range(0, self._root_pow, 2):
            scaled_data = np.sqrt(scaled_data)
            if scaled_labels is not None:
                scaled_labels = np.sqrt(scaled_labels)
        
        transformed_data = data.copy()
        transformed_data[self._scale_features] = scaled_data
        
        return transformed_data if scaled_labels is None else (transformed_data, scaled_labels)
   
class SimpleImputerWrapper(TransformerMixin):

    def __init__(self):
        from sklearn.impute import SimpleImputer
        self._imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    
    def fit(self, data, labels=None):
        self._imputer.fit(data)
        return self
    
    def transform(self, data, labels=None):
        import pandas as pd
        imputed_data = self._imputer.transform(data)
        return pd.DataFrame(imputed_data, columns= data.columns)
        
class PrintData(TransformerMixin, BaseEstimator):
    
    def fit(self, data, labels=None):
        return self
    
    def transform(self, data, labels=None):
        print('Data length: {}, column count: {}'.format(len(data), len(data.columns)))
        print('Data:')
        print(data)
        
        if labels is not None:
            print("Labels:")
            print(labels)
            
        return data if labels is None else (data, labels)
        
class DateFeatureAdder(TransformerMixin, BaseEstimator):
    
    def fit(self, data, labels=None):
        return self
    
    def transform(self, data, labels=None):
        from sklearn.preprocessing import LabelBinarizer
        
        data_copy = data.copy()
        date_feature = get_release_month(data_copy)
        binarizer = LabelBinarizer()
        date_feature_binary = pd.DataFrame(binarizer.fit_transform(date_feature), columns=binarizer.classes_)
        
        data_copy = data_copy.reset_index()
        joint_columns = data_copy.columns.append(date_feature_binary.columns)
        data_copy = pd.concat([data_copy, date_feature_binary], axis=1, ignore_index=True)
        data_copy.columns = joint_columns
        data_copy.set_index('index')
        return data_copy
data, labels = extract_label(training_data_set)
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.3, random_state=1)

print("Training data set samples: {}, testing: {}".format(len(data_train), len(data_test)))
print("Full data set samples: {}.".format(len(data)))
from sklearn import metrics
from sklearn.model_selection import cross_val_score

def model_name(model):
    """ From a given model type extracts its name.
    
    Arguments:
    model -- the model object whose name is needed
    """
    model_name = str(type(model)).split('.')[-1][:-2]
    if model_name == 'Pipeline':
        model_name = model.steps[-1][0]
    return model_name

def model_rmsle(model, data, labels):
    """ Evaluates regression model accuracy using RMSLE error function.
    
    Arguments:
    model -- model to evaluate
    X -- data set to evaluate on
    y -- labels for evaluation
    """
    return np.sqrt(metrics.mean_squared_log_error(labels, np.abs(model.predict(data))))

def model_cv(model, data, labels, scoring='r2'):
    """ Calculates models cross-validationn performance with a given scoring method
    
    Arguments:
    model - model, which performance will be checked
    X - features
    y - labels
    scoring - scoring method (default r2)
    """
    return np.average(cross_val_score(model, data, labels, cv=5, scoring=scoring))

def evaluate_models(models_list, data, labels, cross_val=False):
    """ Evaluates model performance for all models in the list on a given data set (either a test set, or by cross val). 
    Returns a dictionary in such format {model_name: accuracy_score}
    
    Arguments:
    models_list -- list of sklearn models to evaluate
    data_X -- the data for making predictions
    data_y -- labels
    cross_val -- indicator whether to use cross validation
    """
    evaluations = {}
    for i, model in enumerate(models_list):
        evaluations.update({model_name(model) + str(i) : model_cv(model, data, labels) if cross_val else model_rmsle(model, data, labels)})
    
    return evaluations


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import FunctionTransformer

extract_numeric_rows = ('Extract numeric rows', NumericDataSelector())
remove_id = ('Remove id column', FeatureSelector(remove_feature_names=['id']))
impute = ('Impute data', SimpleImputerWrapper())
scale_by_pow_4 = ('Scale data by root of power 4', RootScaler(scale_features=['budget']))
normalize = ('Normalize data', Normalizer())
print_data = ('Print data', PrintData())
month_feature_adder = ('Add binary release month', DateFeatureAdder())

numeric_only_steps = [
    ('Extract numeric rows', NumericDataSelector()),
    ('Remove id column', FeatureSelector(remove_feature_names=['id'])),
    ('Impute data', SimpleImputer(missing_values=np.nan, strategy='median'))
]

numeric_scaled_normalized_steps = [
    ('Extract numeric rows', NumericDataSelector()),
    ('Remove id column', FeatureSelector(remove_feature_names=['id'])),
    ('Impute data', SimpleImputerWrapper()),
    ('Scale data by root of power 4', RootScaler(scale_features=['budget'])),
    ('Normalize data', Normalizer(norm='max'))
]

numeric_and_month = [
    month_feature_adder,
    extract_numeric_rows,
    remove_id,
    impute,
    scale_by_pow_4,
    normalize
]
from sklearn.dummy import DummyRegressor

dummy_regressor = DummyRegressor(strategy='median')
dummy_steps = numeric_only_steps.copy()
dummy_steps.append(('DummyRegressor', dummy_regressor))

dummy_pipeline = Pipeline(steps=dummy_steps)
dummy_pipeline.fit(data_train.copy(), labels_train.copy())
models = [dummy_pipeline]


from sklearn.ensemble import RandomForestRegressor

random_forest = RandomForestRegressor()
random_forest_steps = numeric_only_steps.copy()
random_forest_steps.append(('Random forest regressor', random_forest))

random_forest_pipe = Pipeline(steps=random_forest_steps)
random_forest_pipe.fit(data_train.copy(), labels_train.copy())
models.append(random_forest_pipe)
 

random_forest_scaled_steps = numeric_scaled_normalized_steps.copy()
random_forest_scaled_steps.append(('Random forest scaled & normalized', random_forest))

random_forest_scaled_pipe = Pipeline(steps=random_forest_scaled_steps)
random_forest_scaled_pipe.fit(data_train.copy(), labels_train.copy())
models.append(random_forest_scaled_pipe)


from skopt import BayesSearchCV

random_forest_hyper_search = BayesSearchCV(
    RandomForestRegressor(), 
    {'n_estimators': (10, 100),
     'min_samples_split': (2, 20)},
    n_iter=32)

random_forest_hyper_steps = numeric_scaled_normalized_steps.copy()
random_forest_hyper_steps .append(('Random forest hyper tuning', random_forest_hyper_search))

random_forest_hyper_pipe = Pipeline(steps=random_forest_hyper_steps)
# random_forest_hyper_pipe.fit(data_train.copy(), labels_train.copy())
# models.append(random_forest_hyper_pipe)


random_forest_month_steps = numeric_scaled_normalized_steps.copy()
random_forest_month_steps.append(('Random forest hyper with month', random_forest_hyper_search))

random_forest_month_pipe = Pipeline(steps=random_forest_month_steps)
# random_forest_month_pipe.fit(data_train.copy(), labels_train.copy())
# models.append(random_forest_month_pipe)


from sklearn.svm import SVR

svr = SVR()

support_vector_reg_steps = numeric_only_steps.copy()
support_vector_reg_steps.append(('SupportVectorRegressor', svr))
                       
support_vector_reg_pipe = Pipeline(support_vector_reg_steps)
support_vector_reg_pipe.fit(data_train.copy(), labels_train.copy())
models.append(support_vector_reg_pipe)


svr_normalized_steps = numeric_scaled_normalized_steps.copy()
svr_normalized_steps.append(('SVR normalized', svr))

svr_normalized_pipe = Pipeline(svr_normalized_steps)
svr_normalized_pipe.fit(data_train.copy(), labels_train.copy())
models.append(svr_normalized_pipe)


from skopt import BayesSearchCV

svr_hyper_model = BayesSearchCV(
    SVR(),
    {
        'C': (1e-6, 1e+6, 'log-uniform'),  
        'gamma': (1e-6, 1e+1, 'log-uniform'),
        'degree': (1, 8),
        'kernel': ['linear', 'poly', 'rbf'],
    },
    n_iter=32
)

svr_hyper_steps = numeric_scaled_normalized_steps.copy()
svr_hyper_steps.append(('SVR hyper tuning', svr_hyper_model))

svr_hyper_pipe = Pipeline(svr_hyper_steps)
# svr_hyper_pipe.fit(data_train.copy(), labels_train.copy())
# models.append(svr_hyper_pipe)


svr_hyper_month_steps = numeric_and_month.copy()
svr_hyper_month_steps.append(('SVR hyper & month', svr_hyper_model))

svr_hyper_month_pipe = Pipeline(svr_hyper_month_steps)
# svr_hyper_month_pipe.fit(data_train.copy(), labels_train.copy())
# models.append(svr_hyper_month_pipe)

from xgboost import XGBRegressor

xgbregressor = XGBRegressor()

xgboost_steps = numeric_only_steps.copy()
xgboost_steps.append(('XGBRegressor', xgbregressor))

xgboost_pipe = Pipeline(xgboost_steps)
xgboost_pipe.fit(data_train, labels_train)
models.append(xgboost_pipe)


xgboost_month_steps = numeric_and_month.copy()
xgboost_month_steps.append(('XGBRegressor', xgbregressor))

xgboost_month_pipe = Pipeline(xgboost_month_steps)
# xgboost_month_pipe.fit(data_train, labels_train)
# models.append(xgboost_month_pipe)


evaluations = evaluate_models(models, data_test, labels_test, cross_val=False)

for model, score in evaluations.items():
    print("{} : {}".format(model, score))
    
    
final_model = models[np.argmin(list(evaluations.values()))]
print("Final chosen model is: {}".format(model_name(final_model)))
final_model.fit(data, labels)

predictions = final_model.predict(testing_data_set)
pd.DataFrame(predictions, index=testing_data_set.id, columns=['revenue']).to_csv('./submission.csv')

