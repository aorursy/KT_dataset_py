import os



import numpy as np 

import pandas as pd

from sklearn import model_selection, preprocessing, impute

from sklearn import pipeline, base, compose

from sklearn import feature_selection, ensemble, metrics



import xgboost  



import category_encoders as ce  # feature engineering



import optuna  # hyperparameter optimization



import shap  # model explanations



import matplotlib.pyplot as plt  # plotting



# Adjust plot sizes

plt.rcParams["figure.figsize"] = (15,10)
class AttributeAdder(base.BaseEstimator, base.TransformerMixin):

    """

    Adds age of car and mile per year columns to the dataset.

    Assumes that columns year and mileage are the first two numeric columns in the dataset.

    """

    def __init__(self):

        self.max_year = 2020

        self.column_year = 0

        self.column_mileage = 1

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X):

        age_of_car = self.max_year - X[:, self.column_year]

        mile_per_year = X[:, self.column_mileage] / (age_of_car + 1)

        return np.c_[X[:, 0:self.column_year], age_of_car, 

                     X[:, self.column_year+1:], mile_per_year]
def return_mean_model_prices(x):

    """

    Returns the mean model prices for error analysis

    Used with dataframe.apply

    """

    subset = df[np.logical_and(df.file == x['file'], df.model.str.contains(x['model']))]

    return subset.price.mean()
def model_prediction(X):

    """

    Return model predictions

    

    Arguments:

    ---------

        X (dict): A dictionary with all values

    

    Returns:

    --------

        float: rescaled prediction

    """

    df_pred = pd.DataFrame(X, index=range(len(X['model'])))

    df_pred.mpg = df_pred.mpg.map(lambda x: 200 if x > 200 else x)

    df_pred.model = df_pred.model.map(lambda x: 'rare' if x in rare_models else x)

    

    return np.exp(final_pipeline.predict(df_pred))
df = pd.DataFrame()

headers = ['model', 'year', 'price', 'transmission', 'mileage', 'fuel_size', 'tax', 'mpg', 'engine_size', 'file']

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if not 'unclean' in filename:

            df_new = pd.read_csv(os.path.join(dirname, filename),

                                 names=headers,

                                 skiprows=lambda x: x in [0])

            df_new['file'] = filename[:-4]

            df = df.append(df_new)



# remove the years with 2060 and 1970

# they are either faulty or to different from dataset 

# to include

df = df[~df.year.isin([2060, 1970])]



# Limit the outliers in mpg

df.mpg = df.mpg.map(lambda x: 200 if x > 200 else x)



# map rare models into small groups

rare_models = df.model.value_counts()[df.model.value_counts() <= 20].index.to_list()

df.model = df.model.map(lambda x: 'rare' if x in rare_models else x)



# to have unique indices, appending files repeat indices

df = df.reset_index(drop=True)



X = df.drop('price', axis=1)



# the data is hard to train on since there a lot of sales with exact same 

# price, some noise is added to help the models converge without problems

y = np.log(df[['price']]) + np.random.randn(df.shape[0], 1) * 0.00001



X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,

                                                                    test_size=0.4,

                                                                    random_state=710,

                                                                    stratify=X['file'])



num_types = X_train.select_dtypes('number').columns.to_list()

cat_types = X_train.select_dtypes(exclude='number').columns.to_list()
def model_pipeline(config):

    """

    Create model pipeline according to config dictionary.

    Should include imputer, scaler, n_estimators, learning_rate, reg_alpha, reg_lambda

    

    Designed to accomodate choices made in hyperparameter opt.

    """

    

    # more complicated strategeies yielded worse results

    if config['imputer'] == 'mean':

        imputer = impute.SimpleImputer(strategy='mean')

    elif config['imputer'] == 'median':

        imputer = impute.SimpleImputer(strategy='median')

    else:

        raise ValueError('Wrong Imputer Value')



    # choose scaler, non scaled variables had worse results in previous experiments

    if config['scaler'] == 'standard':

        scaler = preprocessing.StandardScaler()

    elif config['scaler'] == 'power':

        scaler = preprocessing.PowerTransformer()

    else:

        raise ValueError('Wrong Scaler Value')    

    

    

    estimator = xgboost.XGBRegressor(n_estimators=config['n_estimators'],

                                     max_depth=config['max_depth'],

                                     learning_rate=config['learning_rate'],

                                     reg_alpha=config['reg_alpha'],

                                     reg_lambda=config['reg_lambda'],

                                )

    

    # column-wise pipelines for numeric variables

    num_pipeline = pipeline.Pipeline([

        ('imputer', imputer),

        ('attr_adder', AttributeAdder()),

        ('scaler', scaler)

    ])



    # column-wise pipelines for categorical variables

    cat_pipeline = pipeline.Pipeline([

        ('enc', ce.TargetEncoder(handle_unknown='value',

                                 smoothing=config['smoothing'])),

    ])

    

    # combine pipelines

    mid_pipeline = compose.ColumnTransformer([

        ('num', num_pipeline, num_types),

        ('cat', cat_pipeline, cat_types)

    ])

    

    # final pipeline

    full_pipeline = pipeline.Pipeline([

        ('col_trans', mid_pipeline),

        ('regressor', estimator)

    ])

    

    return full_pipeline
def objective(trial):

    """

    Optuna trial function

    """

    X_trial = X_train.copy(deep=True)

    y_trial = y_train.copy(deep=True)

    

    config = {}

    

    config['imputer'] = trial.suggest_categorical('imputer', ['median', 'mean'])

    

    config['smoothing'] = trial.suggest_loguniform('smoothing', 0.01, 1)

    

    config['scaler'] = trial.suggest_categorical('scaler', ['standard', 'power'])



    config['regressor'] = trial.suggest_categorical('regressor', ['RandomForest', 'XGBoost'])

    

    config['n_estimators'] = trial.suggest_int('n_estimators', 350, 500)

    config['max_depth'] = trial.suggest_int('max_depth', 10, 30)        

    config['learning_rate'] = trial.suggest_loguniform('learning_rate', 0.001, 0.1)

    config['reg_alpha'] = trial.suggest_uniform('reg_alpha', 0.5, 1)

    config['reg_lambda'] = trial.suggest_uniform('reg_lambda', 0.5, 1)

        

        

    model = model_pipeline(config)

    

    return model_selection.cross_val_score(model, 

                                           X_trial, 

                                           y_trial, 

                                           n_jobs=-1, 

                                           cv=5).mean()

study = optuna.create_study(direction='maximize')



# higher job numbers crashed the kaggle notebook

study.optimize(objective, n_trials=20, n_jobs=2)



trial = study.best_trial



print('Accuracy: {}'.format(trial.value))

print("Best hyperparameters: {}".format(trial.params))
# retrain with full dataset

final_pipeline = model_pipeline(study.best_params)

final_pipeline.fit(X_train, y_train)
test_preds = final_pipeline.predict(X_test)



print('r2', final_pipeline.score(X_test, y_test))

print('mse', metrics.mean_squared_error(y_test, test_preds))  # please note that the data here is in log scale
# https://github.com/slundberg/shap/issues/1215

# latest shap library has issues with new byte string of the booster

booster = final_pipeline.named_steps['regressor'].get_booster()



model_bytearray = booster.save_raw()[4:]

def replace_booster(self=None):

    return model_bytearray



booster.save_raw = replace_booster
# initiate shap explorer

explainer = shap.TreeExplainer(final_pipeline.named_steps['regressor'])
# get the transformed dataframe for predictions

X_shap = final_pipeline.named_steps['col_trans'].transform(X_train)

samples = np.random.choice(X_shap.shape[0], 500)
shap_values = explainer.shap_values(X_shap[samples])



# add name the name of the new variable to the list

names_features = X_train.columns.tolist() + ['mile_per_year']
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[1,:], X_shap[samples[1],:], feature_names=names_features)
shap.force_plot(explainer.expected_value, shap_values, X_shap[samples], feature_names=names_features)
shap.dependence_plot(6, shap_values, X_shap[samples], feature_names=names_features)
shap.dependence_plot(0, shap_values, X_shap[samples], feature_names=names_features)
shap.summary_plot(shap_values, X_shap[samples], feature_names=names_features)
shap.summary_plot(shap_values, X_shap[samples], feature_names=names_features, plot_type='bar')
# Calculate the error breakdown

analysis_df = pd.DataFrame({'actuals': np.exp(y_test.values).ravel(),

                            'preds': np.exp(test_preds).ravel()}, 

                          index=y_test.index)

analysis_df['error'] = np.abs(analysis_df['preds'] - analysis_df['actuals'])

pd.qcut(analysis_df.error, q=[0, 0.7, 0.95, 0.99, 1]).value_counts()
top_error = analysis_df.sort_values('error', ascending=False).head(10)

df_top_errors = pd.merge(top_error, df, left_index=True, right_index=True)

df_top_errors['mean_model_prices'] = df_top_errors.apply(lambda x: return_mean_model_prices(x), axis=1)

df_top_errors
# totally nonsense models sent

new_input = {'model': ['2 Series', 'A8'],

             'year': [2017, 2019], 

             'transmission': ['Automatic', 'Semi-Auto'],

             'mileage': [2000, 30000], 

             'fuel_size': ['Diesel', 'Petrol'],

             'tax': [20, 145],

             'mpg': [33.2, 11.0],

             'engine_size': [3.0, 5.5],

             'file': ['audi', 'merc']}
model_prediction(new_input)