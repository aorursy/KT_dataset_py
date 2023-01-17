from __future__ import annotations
from typing import List, Dict, Tuple, Callable, Iterable

from enum import Enum
from itertools import product
import numpy as np
import pandas as pd
from pandas.core.groupby import SeriesGroupBy, DataFrameGroupBy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import TransformerMixin, BaseEstimator
df = pd.read_csv('../input/electric-motor-temperature/pmsm_temperature_data.csv')
df.head(5)
df.tail(5)
df[df.isna().any(axis=1)]
df.dropna(inplace=True)
df.describe().T
MEASURES_IN_SECOND = 2
SECONDS_IN_HOUR = 3600

PLOT_STYLE = "darkgrid"
PLOT_PRIMARY_COLOR = "#333333"
PLOT_SECONDARY_COLOR = "#cc9602"

TEMPERATURE_FEATURES = ['pm', 'stator_yoke', 'stator_tooth', 'stator_winding']
TARGET_FEATURES = TEMPERATURE_FEATURES + ["torque"]
def plot_profile_durations(df: pd.DataFrame) -> None:
    '''
    - get DataFrame of dataset
    - groupes it by 'profile_id' column
    - calculate durations for each profile_id 
    - plot barplot of profile durations

    '''

    grouped_df = (df.groupby('profile_id')
                    .size()
                    .rename('samples')
                    .reset_index())
    
    with sns.axes_style(PLOT_STYLE):
        _, ax = plt.subplots(1, 1, figsize=(20,3))
        sns.barplot(y='samples', x='profile_id', data=grouped_df, color=PLOT_PRIMARY_COLOR, ax=ax)
        
        hours_ticks: np.ndarray = np.arange(1, 8)
        
        ax.set_yticks(MEASURES_IN_SECOND*SECONDS_IN_HOUR*hours_ticks)
        ax.set_yticklabels([f'{h}' for h in hours_ticks])
        ax.set_ylim((0*MEASURES_IN_SECOND*SECONDS_IN_HOUR, 7*MEASURES_IN_SECOND*SECONDS_IN_HOUR))
        ax.set_ylabel('duration, hours')        
plot_profile_durations(df)
def plot_profile_time_series(df: pd.DataFrame, 
                             profile_id: int, 
                             num_of_points: int = None, 
                             features: List[str]=[]) -> None:
    '''
    - get:
      * DataFrame of dataset, 
      * single profile_id, 
      * number of points to plot from the begining of profile, 
      * list of features to plot 
    - plot timeseries of profile (each feature at single axes)

    '''

    if num_of_points is None:
        filtered_df=df[features][df.profile_id == profile_id].reset_index()
    else:
        filtered_df=df[features][df.profile_id == profile_id].reset_index()[0:num_of_points]

    with sns.axes_style(PLOT_STYLE):
        _, axes = plt.subplots(len(features), 1, figsize=(15, len(features)*2), sharex=True)
        plt.xlabel("sample")

        for ax, feature_name in zip(axes, features):
            ax.plot(filtered_df[feature_name], color=PLOT_PRIMARY_COLOR)
            ax.set_ylabel(feature_name)
def plot_profile_time_series_single_grid(df: pd.DataFrame, 
                                         profile_id: int, 
                                         num_of_points: int = None, 
                                         features: List[str]=[]) -> None:
  
    '''
    - get:
      * DataFrame of dataset, 
      * single profile_id, 
      * number of points to plot from the begining of profile, 
      * list of features to plot
    - plot timeseries of profile (on the single axes)

    '''

    if num_of_points is None:
        filtered_df=df[features][df.profile_id == profile_id].reset_index()
    else:
        filtered_df=df[features][df.profile_id == profile_id].reset_index()[0:num_of_points]

    with sns.axes_style(PLOT_STYLE):
        _, ax = plt.subplots(1, 1, figsize=(15, 2))
        for idx, feature in enumerate(features):
            if idx == 0:
                ax.plot(filtered_df[feature], color=PLOT_PRIMARY_COLOR, label=feature)
            elif idx == 1:
                ax.plot(filtered_df[feature], color=PLOT_SECONDARY_COLOR, label=feature)
            else:
                ax.plot(filtered_df[feature], label=feature)
        ax.set_xlabel('sample')
        ax.legend()
# example of short-time motor running
plot_profile_time_series(df, 46, features = ['pm','coolant','ambient', 'torque', 'i_d'])
# example of long-time motor running
plot_profile_time_series(df, 20, features=['pm','coolant','ambient', 'torque', 'i_d'])
def calculate_magnitude(value_d: pd.Series, value_q: pd.Series) -> pd.Series:
    '''
    Returns magnitude of parameter represented by d- and -q axis values.
    
    '''
    return np.sqrt(value_d**2 + value_q**2)


def calculate_apparent_power(current: pd.Series, voltage: pd.Series) -> pd.Series:
    '''
    Returns apparent power calculated with current and voltage values.
    
    '''
    return current * voltage
    

def calculate_active_power(current_d: pd.Series, 
                           current_q: pd.Series, 
                           voltage_d: pd.Series, 
                           voltage_q: pd.Series) -> pd.Series:

    '''
    Returns active power calculated with d- and -q axis values.
    
    '''
    return current_d * voltage_d + current_q * voltage_q
    

def calculate_reactive_power(current_d: pd.Series, 
                             current_q: pd.Series, 
                             voltage_d: pd.Series, 
                             voltage_q: pd.Series) -> pd.Series:
    '''
    Returns reactive power calculated with d- and -q axis values.
    
    '''
    return current_d * voltage_q - current_q * voltage_d
class StatorParametersTransformer(TransformerMixin):
    '''
    Custom sklearn feature Transfomer for adding stator parameters values
    
    '''

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> StatorParametersTransformer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        '''
        Get input DataFrame and return new DataFrame with added stator parameters
        
        '''
        return X.assign(**{
          'current': lambda x: calculate_magnitude(x["i_d"], x["i_q"]),
          'voltage': lambda x: calculate_magnitude(x["u_d"], x["u_q"]),
          'apparent_power': lambda x: calculate_apparent_power(x["current"], x["voltage"]),
          'active_power': lambda x: calculate_active_power(x["i_d"], x["i_q"], x["u_d"], x["u_q"]),
          'reactive_power': lambda x: calculate_reactive_power(x["i_d"], x["i_q"], x["u_d"], x["u_q"]),
        })
df_with_stator_parameters = StatorParametersTransformer().fit_transform(df)
plot_profile_time_series(df_with_stator_parameters, 46, features=['pm','current'])
plot_profile_time_series(df_with_stator_parameters, 20, features=['pm','current'])
def groupby_and_extract(column: str, feature: str) -> Callable:
    '''
    Decorator function that get single column and single feature names
    and returns actual decorator that apply groupby column operation to input DataFrame
    and select feature column operation

    Usage:
    @groupby_and_extract("profile_id", feature_name)
        def calculate_ewma(series: SeriesGroupBy):
          ...
    '''
    def decorator(func: Callable) -> Callable:
        def wrapper(df: pd.DataFrame) -> DataFrameGroupBy:
            return func(df.groupby("profile_id")[feature])
        return wrapper
    return decorator
  

def define_ewma_calculation_func(feature_name: str, span: int) -> Callable:
    '''
    Get feature name and span
    Returns function that will be called to calculate EWMA of feature 
    with span in assing method of DataFrame

    '''

    # TODO: Fine and clear, but group by for each new feature wired and not efficient. 
    # Need refactoring for multicolumn EWMA.
    @groupby_and_extract("profile_id", feature_name)
    def calculate_ewma(series: SeriesGroupBy) -> pd.Series:
        transformed_series: pd.Series = series.transform(lambda x: x.ewm(span, min_periods=1).mean())
        return transformed_series.reset_index(drop=True)
      
    return calculate_ewma


def define_ewma_features(features: List[str], spans: List[int]) -> Dict[str, Callable]:
    '''
    Get feature names and spans
    Returns dictionary with new column names as keys and functions 
    to generate EWMA features as items

    '''
    result: Dict[str, Callable] = dict()

    for feature, span in product(features, spans):
        feature_name: str = f"{feature}_ewma_{span}"
        calcualtion_func: Callable = define_ewma_calculation_func(feature, span)
        result[feature_name] = calcualtion_func

    return result
class EwmaTransformer(BaseEstimator, TransformerMixin):
    '''
    Custom sklearn feature Transfomer for adding EWMA features
    
    '''
    def __init__(self, 
                 columns: List[str] = [], 
                 spans: List[int] = [], 
                 drop_transformed: bool = True):
      
        self.columns = columns
        self.spans = spans
        self.drop_transformed = drop_transformed

    def set_params(self, **params):
        self.columns = params["columns"]
        self.spans = params["spans"]
        self.drop_transformed = params["drop_transformed"]

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> EwmaTransformer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        '''
        Get input DataFrame and return new DataFrame with added EWMA features
        
        '''
        ewma_features: Dict[str, Callable] = define_ewma_features(self.columns, self.spans)
        return X.assign(**ewma_features).drop(self.columns if self.drop_transformed else [], axis=1)
class Debug(TransformerMixin):
    '''
    Custom sklearn feature Transfomer for debug Pipline steps
    
    '''
    def __init__(self, title: str):
      self.title = title

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        print(self.title)
        print(f"Total shape: {X.shape}")
        print(f"NaN shape{X[X.isna().any(axis=1)].shape}\n\n")
        return X

    def fit(self, X: pd.DataFrame, y: pd.Series = None, **fit_params) -> Debug:
        return self
ewma_spans: List[int] = [1300, 3000, 5000]
ewma_columns: List[str] = ['ambient', 'coolant', 'u_d', 'u_q', 
                            'motor_speed', 'i_d','i_q', 'current', 'voltage', 
                            'apparent_power', 'active_power', 'reactive_power']

feature_tarnsformation_pipeline = make_pipeline(
    StatorParametersTransformer(),
    EwmaTransformer(ewma_columns, ewma_spans, drop_transformed=False),
)

new_features_df = feature_tarnsformation_pipeline.fit_transform(df)
plot_profile_time_series_single_grid(new_features_df.reset_index(), 20, features=['current', 'current_ewma_1300'])
plot_profile_time_series_single_grid(new_features_df.reset_index(), 20, features=['pm', 'current_ewma_1300'])
plot_profile_time_series_single_grid(new_features_df.reset_index(), 20, features=['ambient', 'ambient_ewma_1300'])
# Note that profile_id is needed for tests and evaluation, 
# so we will need to deal with it on the modelling steps
# maybe I should solved with moving this column to index

X = new_features_df.drop(columns=TARGET_FEATURES + ["profile_id"])
y = new_features_df["pm"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# Use SelectKBest out of feature_tarnsformation_pipeline because of DataFrame interface
# We can change k and see how it influence on results
# K='all' is for all features selected

selector = SelectKBest(f_regression, k='all')
selector.fit_transform(X_train, y_train)
mask = selector.get_support()
selected_features = X_train.columns[mask]

X_train_filtered = X_train[selected_features]
X_test_filtered = X_test[selected_features]

selected_features
model_pipline = Pipeline([
  ("regressor", Ridge())
])

params_grid = {
    "regressor__alpha": np.logspace(-8, 3, num=12, base=10),
    "regressor__fit_intercept": [True, False],
}

model = GridSearchCV(model_pipline, 
                     params_grid, 
                     scoring=make_scorer(r2_score), 
                     n_jobs=-1,
                     cv=10,
                     verbose=1,
                     refit=True,
                     return_train_score=True
                     )

model.fit(X_train_filtered, y_train)

cv_results_df = pd.DataFrame(model.cv_results_)
cv_results_df[["param_regressor__alpha", "param_regressor__fit_intercept", 
               "mean_train_score","std_train_score",
               "mean_test_score","std_test_score", 
               "rank_test_score"]]
model.best_estimator_.named_steps['regressor'].coef_
def evaluate_test(estimator: BaseEstimator, 
                  score: Callable, 
                  X_test: pd.DataFrame, 
                  y_test: pd.Series) -> float:
    '''
    Returns specified score for estimator and input data

    '''

    y_pred = estimator.predict(X_test)
    return score(y_test, y_pred)
def score_on_profiles(df: pd.DataFrame, 
                      estimator: BaseEstimator, 
                      score: Callable, 
                      selected_features: Iterable, 
                      profile_ids: List[str]=None) -> List[Tuple]:
    '''
    Returns specified scores for estimator and input data
    Use selected_features for indexing data for concrete estimator

    '''
    profile_ids_to_score = df.profile_id.unique() if profile_ids is None else profile_ids
    
    result = list()
    for profile_id in profile_ids_to_score:
        df_profile = df[df.profile_id == profile_id]
        profile_score = evaluate_test(estimator, score, df_profile[selected_features], df_profile["pm"])
        result.append((int(profile_id), profile_score))

    return result
# df should have profile_id
def plot_fitted_values(df: pd.DataFrame, 
                       estimator: BaseEstimator,
                       selected_features: Iterable, 
                       profile_ids: List[str]=None) -> None:

    '''
    Plot true and fitted values for estimator and input data for specified profile_id
    Use selected_features for indexing data for concrete estimator

    '''
    profile_ids_to_plot = df.profile_id.unique() if profile_ids is None else profile_ids

    with sns.axes_style(PLOT_STYLE):
        _, axes = plt.subplots(len(profile_ids_to_plot), 
                              1, 
                              figsize=(15, len(profile_ids_to_plot)*2))
        plt.xlabel("sample")

        for ax, profile_id in zip(axes, profile_ids_to_plot):
            df_profile = df[df.profile_id == profile_id]
            y_pred = estimator.predict(df_profile[selected_features])
            y_true = df_profile["pm"].values

            ax.plot(y_true, color=PLOT_PRIMARY_COLOR, label="true")
            ax.plot(y_pred, color=PLOT_SECONDARY_COLOR, label="prediction")
            ax.legend()
            ax.set_ylabel(f"pm: pid {profile_id}")
score = evaluate_test(model.best_estimator_, r2_score, X_test_filtered, y_test)
f"Evaluation on test dataset result: {score: 0.3f}"
scores = score_on_profiles(new_features_df, model.best_estimator_, r2_score, selected_features)
scores_df = pd.DataFrame(scores, columns=["profile_id", "score"]).sort_values(by=['score'], ascending=False).reset_index(drop=True)
scores_df
# good fittings:
best_5_fitted_profiles_id = scores_df.loc[0:5, "profile_id"].values
plot_fitted_values(new_features_df, model.best_estimator_, selected_features, best_5_fitted_profiles_id)
# worst fittings:
worst_5_fitted_profiles_id = scores_df.iloc[-5:]["profile_id"].values
plot_fitted_values(new_features_df, model.best_estimator_, selected_features, worst_5_fitted_profiles_id)