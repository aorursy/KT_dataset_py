# Imports

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('fivethirtyeight')

import warnings

warnings.filterwarnings("ignore")

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, LSTM

from tensorflow.keras.callbacks import EarlyStopping

from lightgbm import LGBMRegressor

import time

from sklearn.model_selection import cross_val_score
# Loading data

df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')

df_train.head()
df_train['Date'] = pd.to_datetime(df_train['Date'], infer_datetime_format=True)

df_test['Date'] = pd.to_datetime(df_test['Date'], infer_datetime_format=True)

df_intersection = df_test[df_test['Date'] <= np.max(df_train['Date'])]

df_intersection
# Following the idea at

# https://www.kaggle.com/ranjithks/25-lines-of-code-results-better-score#Fill-NaN-from-State-feature

# Filling NaN states with the Country



EMPTY_VAL = "EMPTY_VAL"



def fillState(state, country):

    if state == EMPTY_VAL: return country

    return state



df_train['Province_State'].fillna(EMPTY_VAL, inplace=True)

df_train['Province_State'] = df_train.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)



df_test['Province_State'].fillna(EMPTY_VAL, inplace=True)

df_test['Province_State'] = df_test.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)



df_intersection['Province_State'].fillna(EMPTY_VAL, inplace=True)

df_intersection['Province_State'] = df_intersection.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)



df_intersection.tail()
# Adding validation data into the Intersection DF

states = sorted(set(df_intersection['Province_State']))

df_intersection['ConfirmedCases'] = float('NaN')

df_intersection['Fatalities'] = float('NaN')



for state in states:

    dates = sorted(set(df_intersection[df_intersection['Province_State'] == state]['Date']))

    min_date = np.min(dates)

    max_date = np.max(dates)

    idx = df_intersection[df_intersection['Province_State'] == state].index

    values = df_train[(df_train['Province_State'] == state) & (df_train['Date'] >= min_date) & (df_train['Date'] <= max_date)][['ConfirmedCases', 'Fatalities']].values

    values = pd.DataFrame(values, index = list(idx), columns=['ConfirmedCases', 'Fatalities'])

    df_intersection['ConfirmedCases'].loc[idx] = values['ConfirmedCases']

    df_intersection['Fatalities'].loc[idx] = values['Fatalities']

df_intersection
# Filtering data for public leaderboard

df_train = df_train[df_train['Date'] < np.min(df_test['Date'])]

# Check if any Province_State value on test dataset isn't on train dataset

# If nothing prints, everything is okay

for a in set(df_test['Province_State']):

    if a not in set(df_train['Province_State']):

        print (a)
# Making Date become timestamp

df_train['Date'] = df_train['Date'].apply(lambda s: time.mktime(s.timetuple()))

df_test['Date'] = df_test['Date'].apply(lambda s: time.mktime(s.timetuple()))

df_intersection['Date'] = df_intersection['Date'].apply(lambda s: time.mktime(s.timetuple()))



min_timestamp = np.min(df_train['Date'])

df_train['Date'] = df_train['Date'].apply(lambda s: (s - min_timestamp) / 86400.0)

df_test['Date'] = df_test['Date'].apply(lambda s: (s - min_timestamp) / 86400.0)

df_intersection['Date'] = df_intersection['Date'].apply(lambda s: (s - min_timestamp) / 86400.0)
# Generating features based on evolution of COVID-19

# Idea from https://www.kaggle.com/binhlc/sars-cov-2-exponential-model-week-2

evolution = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

def generateFeatures (state):

    should_filter = False

    train = df_train[df_train['Province_State'] == state].drop(columns=['Id'])

    test  = df_test[df_test['Province_State'] == state].drop(columns=['ForecastId'])

    y_cases = train['ConfirmedCases']

    y_fatal = train['Fatalities']

    for evo_type in ['ConfirmedCases', 'Fatalities']:

        for value in evolution:

            min_day = train[train[evo_type] >= value]['Date']

            if min_day.count() > 0:

                min_day = np.min(min_day)

                should_filter = True

            else:

                print ("{} -> Not found min_day for {} {}".format(state, evo_type, value))

                continue

            train['{}_{}'.format(evo_type, value)] = train['Date'].apply(lambda x: x - min_day)

            test ['{}_{}'.format(evo_type, value)] = test ['Date'].apply(lambda x: x - min_day)

    train.drop(columns=['ConfirmedCases', 'Fatalities', 'Province_State', 'Country_Region'], inplace=True)

    test.drop(columns=['Province_State', 'Country_Region'], inplace=True)

    if should_filter:

        idx     = train[train['ConfirmedCases_1'] >= 0].index

        train   = train.loc[idx]

        y_cases = y_cases.loc[idx]

        y_fatal = y_fatal.loc[idx]

    return train, test, y_cases, y_fatal



dataframes = {}

states = sorted(set(df_train['Province_State']))

for state in states:

    dataframes[state] = {}

    train, test, y_cases, y_fatal = generateFeatures(state)

    dataframes[state]['train']   = train

    dataframes[state]['test']    = test

    dataframes[state]['y_cases'] = y_cases

    dataframes[state]['y_fatal'] = y_fatal
# Checking shapes

state = 'Georgia'

print (dataframes[state]['train'].shape, dataframes[state]['test'].shape, dataframes[state]['y_cases'].shape, dataframes[state]['y_fatal'].shape)

dataframes[state]['test'].head()
from tqdm import tqdm

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression, BayesianRidge, Lasso

from sklearn.metrics import mean_squared_log_error

from sklearn.base import TransformerMixin

from sklearn.datasets import make_regression

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.ensemble.weight_boosting import AdaBoostRegressor

from sklearn.linear_model.base import LinearRegression

from sklearn.linear_model.passive_aggressive import PassiveAggressiveRegressor

from sklearn.linear_model.theil_sen import TheilSenRegressor

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone



def handle_predictions (predictions, lowest = 0):

    #predictions = np.round(predictions, 0)

    # Predictions can't be negative

    predictions[predictions < 0] = 0

    # Predictions can't decrease from greatest value on train dataset

    predictions[predictions < lowest] = lowest

    # Predictions can't decrease over time

    for i in range(1, len(predictions)):

        if predictions[i] < predictions[i - 1]:

            predictions[i] = predictions[i - 1]

    #return predictions.astype(int)

    return predictions



def fillSubmission (state, column, values,):

    idx = df_test[df_test['Province_State'] == state].index

    values = pd.DataFrame(np.array(values), index = list(idx), columns=[column])

    submission[column].loc[idx] = values[column]

    return submission



def avg_rmsle():

    idx = df_intersection.index

    my_sub = submission.loc[idx][['ConfirmedCases', 'Fatalities']]

    cases_pred = my_sub['ConfirmedCases'].values

    fatal_pred = my_sub['Fatalities'].values

    pred = np.append(cases_pred, fatal_pred)

    cases_targ = df_intersection.loc[idx]['ConfirmedCases'].values

    fatal_targ = df_intersection.loc[idx]['Fatalities'].values

    targ = np.append(cases_targ, fatal_targ)

    score = np.sqrt(mean_squared_log_error( targ, pred ))

    return score



class CustomEnsemble (BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, models, meta_model):

        self.models = models

        self.meta_model = meta_model

    def fit(self,X,y):

        predictions = np.zeros((X.shape[0], len(self.models)))

        for i, model in enumerate(self.models):

            model.fit (X, y)

            predictions[:,i] = model.predict(X)

        self.meta_model.fit(predictions, y)

    def predict(self,X):

        predictions = np.zeros((X.shape[0], len(self.models)))

        for i, model in enumerate(self.models):

            predictions[:,i] = model.predict(X)

        return self.meta_model.predict(predictions)

    def __str__ (self):

        return "<CustomEnsemble (meta={}, models={})>".format(self.meta_model, self.models)



def make_combinations (iterable):

    from itertools import combinations

    my_combs = []

    for item in iterable.copy():

        iterable.remove(item)

        for i in range(len(iterable)):

            for comb in combinations(iterable, i+1):

                my_combs.append((item, comb))

        iterable.append(item)

    return my_combs



test_models = [

    make_pipeline(PolynomialFeatures(2), LinearRegression()),              # 0.8990346097108978

    make_pipeline(PolynomialFeatures(2), TheilSenRegressor()),             # 0.8910456039208402

    make_pipeline(PolynomialFeatures(2), BayesianRidge()),                 # 0.8997409933399905

    make_pipeline(PolynomialFeatures(2), Lasso()),                         # 0.8920475587104756

]



# for model in test_models:

#     print (' * Model: {}'.format(model))

#     for state in states:

#         train   = dataframes[state]['train']

#         test    = dataframes[state]['test']

#         y_cases = dataframes[state]['y_cases']

#         y_fatal = dataframes[state]['y_fatal']

#         model.fit(train, y_cases)

#         cases = model.predict(test)

#         lowest_pred = np.max(df_train[df_train['Province_State'] == state]['ConfirmedCases'].values)

#         cases = handle_predictions(cases, lowest_pred)

#         submission = fillSubmission (state, 'ConfirmedCases', cases)

#         model.fit(train, y_fatal)

#         fatal = model.predict(test)

#         lowest_pred = np.max(df_train[df_train['Province_State'] == state]['Fatalities'].values)

#         fatal = handle_predictions(fatal, lowest_pred)

#         submission = fillSubmission (state, 'Fatalities', fatal)

#     print ('   - Score: {}'.format(avg_rmsle()))

my_combs = make_combinations(test_models)

best = 10000

results = []

with tqdm(total = len(my_combs) * len(states)) as pbar:

    for comb in my_combs:

        for state in states:

            train   = dataframes[state]['train']

            test    = dataframes[state]['test']

            y_cases = dataframes[state]['y_cases']

            y_fatal = dataframes[state]['y_fatal']

            model = CustomEnsemble(list(comb[1]), comb[0])

            model.fit(train, y_cases)

            cases = model.predict(test)

            lowest_pred = np.max(df_train[df_train['Province_State'] == state]['ConfirmedCases'].values)

            cases = handle_predictions(cases, lowest_pred)

            submission = fillSubmission (state, 'ConfirmedCases', cases)

            model.fit(train, y_fatal)

            fatal = model.predict(test)

            lowest_pred = np.max(df_train[df_train['Province_State'] == state]['Fatalities'].values)

            fatal = handle_predictions(fatal, lowest_pred)

            submission = fillSubmission (state, 'Fatalities', fatal)

            pbar.update(1)

        score = avg_rmsle()

        results.append(score)

        if (score < best):

            print ("Score {:.4f} is better than previous best. Saving...".format(score))

            best = score

            best_model = model
# Making predicitons using the best model for the private leaderboard

# Load raw train

df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

# Handle it

df_train['Date'] = pd.to_datetime(df_train['Date'], infer_datetime_format=True)

EMPTY_VAL = "EMPTY_VAL"

df_train['Province_State'].fillna(EMPTY_VAL, inplace=True)

df_train['Province_State'] = df_train.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)

df_train['Date'] = df_train['Date'].apply(lambda s: time.mktime(s.timetuple()))

df_train['Date'] = df_train['Date'].apply(lambda s: (s - min_timestamp) / 86400.0)

# Re-generate features and predict

dataframes = {}

states = sorted(set(df_train['Province_State']))

for state in states:

    dataframes[state] = {}

    train, test, y_cases, y_fatal = generateFeatures(state)

    model = best_model

    model.fit(train, y_cases)

    cases = model.predict(test)

    lowest_pred = np.max(df_train[df_train['Province_State'] == state]['ConfirmedCases'].values)

    cases = handle_predictions(cases, lowest_pred)

    submission = fillSubmission (state, 'ConfirmedCases', cases)

    model.fit(train, y_fatal)

    fatal = model.predict(test)

    lowest_pred = np.max(df_train[df_train['Province_State'] == state]['Fatalities'].values)

    fatal = handle_predictions(fatal, lowest_pred)

    submission = fillSubmission (state, 'Fatalities', fatal)

def checkState (state):

    idx = df_test[df_test['Province_State'] == state].index

    return submission.loc[idx]



def plotStatus (states):

    plt.figure(figsize=(14,8))

    plt.title('COVID-19 on {}'.format(states))

    if type(states) == list:

        legend = []

        for state in states:

            df = df_train[df_train['Province_State'] == state]

            idx = df_test[df_test['Province_State'] == state].index

            plt.xlabel('#Days since dataset')

            plt.ylabel('Number')

            plt.plot(df['Date'], df['ConfirmedCases'])

            plt.plot(range(int(np.max(df['Date'])), int(np.max(df['Date'])) + len(idx)), submission['ConfirmedCases'].loc[idx])

            plt.plot(df['Date'], df['Fatalities'])

            plt.plot(range(int(np.max(df['Date'])), int(np.max(df['Date'])) + len(idx)), submission['Fatalities'].loc[idx])

            legend.append('{} confirmed cases'.format(state))

            legend.append('{} predicted cases'.format(state))

            legend.append('{} fatalities'.format(state))

            legend.append('{} predicted fatalities'.format(state))

        plt.legend(legend)

    else:

        state = states

        df = df_train[df_train['Province_State'] == state]

        plt.figure(figsize=(14,8))

        plt.xlabel('#Days since dataset')

        plt.ylabel('Number')

        plt.plot(df['Date'], df['ConfirmedCases'])

        plt.plot(df['Date'], df['Fatalities'])

        plt.legend(['Confirmed cases', 'Fatalities'])

    plt.show()



raw_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

raw_test  = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

raw_train['Date'] = pd.to_datetime(raw_train['Date'], infer_datetime_format=True)

raw_test['Date']  = pd.to_datetime(raw_test['Date'], infer_datetime_format=True)



def rmsle (state):

    idx = df_intersection[df_intersection['Province_State'] == state].index

    my_sub = submission.loc[idx][['ConfirmedCases', 'Fatalities']]

    cases_pred = my_sub['ConfirmedCases'].values

    fatal_pred = my_sub['Fatalities'].values

    cases_targ = df_intersection.loc[idx]['ConfirmedCases'].values

    fatal_targ = df_intersection.loc[idx]['Fatalities'].values

    cases = np.sqrt(mean_squared_log_error( cases_targ, cases_pred ))

    fatal = np.sqrt(mean_squared_log_error( fatal_targ, fatal_pred ))

    return cases, fatal



plotStatus(['Zhejiang'])
for state in states:

    print ("  * {}".format(state))

    scores = rmsle(state)

    print ("    - {}\n    - {}".format(scores[0], scores[1]))
submission.to_csv('submission.csv', index=False)
#

#  FUTURE:

#  - Get datasets with no confirmed cases and try to implement other features (country relationships maybe? continent? distance between countries?)

#  - Tune more models

#  - Check this out: https://stats.stackexchange.com/questions/139042/ensemble-of-different-kinds-of-regressors-using-scikit-learn-or-any-other-pytho

#