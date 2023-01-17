import numpy as np

import pandas as pd

from pandas_datareader import data



from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

import xgboost as xgb

from sklearn.metrics import classification_report



# Suppress warnings

import warnings

warnings.filterwarnings('ignore')
# Custom function to pull prices of stocks

def get_price_var(symbol):

    '''

    Get historical price data for a given symbol leveraging the power of pandas_datareader and Yahoo.

    Compute the difference between first and last available time-steps in terms of Adjusted Close price.

    

    Input: ticker symbol

    Output: percent price variation 

    '''

    # read data

    prices = data.DataReader(symbol, 'yahoo', '2019-01-01', '2019-12-31')['Adj Close']



    # get all timestamps for specific lookups

    today = prices.index[-1]

    start = prices.index[0]



    # calculate percentage price variation

    price_var = ((prices[today] - prices[start]) / prices[start]) * 100

    return price_var
# Load 2019 percent price variation for all tech stocks

PVAR = pd.read_csv('../input/Example_2019_price_var.csv', index_col=0)



# Load dataset with all financial indicators (referring to end of 2018)

DATA = pd.read_csv('../input/Example_DATASET.csv', index_col=0)
# Divide data in train and testing splits

train_split, test_split = train_test_split(DATA, test_size=0.2, random_state=1, stratify=DATA['class'])

X_train = train_split.iloc[:, :-1].values

y_train = train_split.iloc[:, -1].values

X_test = test_split.iloc[:, :-1].values

y_test = test_split.iloc[:, -1].values



print(f'Total number of samples: {DATA.shape[0]}')

print()

print(f'Number of training samples: {X_train.shape[0]}')

print()

print(f'Number of testing samples: {X_test.shape[0]}')

print()

print(f'Number of features: {X_train.shape[1]}')
# Standardize input data

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)
# Parameter grid to be tuned

tuned_parameters = [{'kernel': ['rbf', 'linear'], 'gamma': [1e-3, 1e-4], 'C': [0.01, 0.1, 1, 10, 100]}]



clf1 = GridSearchCV(SVC(random_state=1),

                    tuned_parameters,

                    n_jobs=4,

                    scoring='precision_weighted',

                    cv=5)



clf1.fit(X_train, y_train)



print('Best score and parameters found on development set:')

print()

print('%0.3f for %r' % (clf1.best_score_, clf1.best_params_))

print()
# Parameter grid to be tuned

tuned_parameters = {'n_estimators': [1024, 4096],

                    'max_features': ['auto', 'sqrt'],

                    'max_depth': [4, 6, 8],

                    'criterion': ['gini', 'entropy']}



clf2 = GridSearchCV(RandomForestClassifier(random_state=1),

                    tuned_parameters,

                    n_jobs=4,

                    scoring='precision_weighted',

                    cv=5)



clf2.fit(X_train, y_train)



print('Best score and parameters found on development set:')

print()

print('%0.3f for %r' % (clf2.best_score_, clf2.best_params_))

print()
# Parameter grid to be tuned

tuned_parameters = {'learning_rate': [0.01, 0.001],

                    'max_depth': [4, 6, 8],

                    'n_estimators': [512, 1024]}



clf3 = GridSearchCV(xgb.XGBClassifier(random_state=1),

                   tuned_parameters,

                   n_jobs=4,

                   scoring='precision_weighted', 

                   cv=5)



clf3.fit(X_train, y_train)



print('Best score and parameters found on development set:')

print()

print('%0.3f for %r' % (clf3.best_score_, clf3.best_params_))

print()
# Parameter grid to be tuned

tuned_parameters = {'hidden_layer_sizes': [(32,), (64,), (32, 64, 32)],

                    'activation': ['tanh', 'relu'],

                    'solver': ['lbfgs', 'adam']}



clf4 = GridSearchCV(MLPClassifier(random_state=1, batch_size=4, early_stopping=True), 

                    tuned_parameters,

                    n_jobs=4,

                    scoring='precision_weighted',

                    cv=5)



clf4.fit(X_train, y_train)



print('Best score, and parameters, found on development set:')

print()

print('%0.3f for %r' % (clf4.best_score_, clf4.best_params_))

print()
# Get 2019 price variations ONLY for the stocks in testing split

pvar_test = PVAR.loc[test_split.index.values, :]
# Initial investment can be $100 for each stock whose predicted class = 1

buy_amount = 100



# In new dataframe df1, store all the information regarding each model's predicted class and relative gain/loss in $USD

df1 = pd.DataFrame(y_test, index=test_split.index.values, columns=['ACTUAL']) # first column is the true class (BUY/INGORE)



df1['SVM'] = clf1.predict(X_test) # predict class for testing dataset

df1['VALUE START SVM [$]'] = df1['SVM'] * buy_amount # if class = 1 --> buy $100 of that stock

df1['VAR SVM [$]'] = (pvar_test['2019 PRICE VAR [%]'].values / 100) * df1['VALUE START SVM [$]'] # compute price variation in $

df1['VALUE END SVM [$]'] = df1['VALUE START SVM [$]'] + df1['VAR SVM [$]'] # compute final value



df1['RF'] = clf2.predict(X_test)

df1['VALUE START RF [$]'] = df1['RF'] * buy_amount

df1['VAR RF [$]'] = (pvar_test['2019 PRICE VAR [%]'].values / 100) * df1['VALUE START RF [$]']

df1['VALUE END RF [$]'] = df1['VALUE START RF [$]'] + df1['VAR RF [$]']



df1['XGB'] = clf3.predict(X_test)

df1['VALUE START XGB [$]'] = df1['XGB'] * buy_amount

df1['VAR XGB [$]'] = (pvar_test['2019 PRICE VAR [%]'].values / 100) * df1['VALUE START XGB [$]']

df1['VALUE END XGB [$]'] = df1['VALUE START XGB [$]'] + df1['VAR XGB [$]']



df1['MLP'] = clf4.predict(X_test)

df1['VALUE START MLP [$]'] = df1['MLP'] * buy_amount

df1['VAR MLP [$]'] = (pvar_test['2019 PRICE VAR [%]'].values / 100) * df1['VALUE START MLP [$]']

df1['VALUE END MLP [$]'] = df1['VALUE START MLP [$]'] + df1['VAR MLP [$]']



# Show dataframe df1

df1.head()
# Create a new, compact, dataframe in order to show gain/loss for each model

start_value_svm = df1['VALUE START SVM [$]'].sum()

final_value_svm = df1['VALUE END SVM [$]'].sum()

net_gain_svm = final_value_svm - start_value_svm

percent_gain_svm = (net_gain_svm / start_value_svm) * 100



start_value_rf = df1['VALUE START RF [$]'].sum()

final_value_rf = df1['VALUE END RF [$]'].sum()

net_gain_rf = final_value_rf - start_value_rf

percent_gain_rf = (net_gain_rf / start_value_rf) * 100



start_value_xgb = df1['VALUE START XGB [$]'].sum()

final_value_xgb = df1['VALUE END XGB [$]'].sum()

net_gain_xgb = final_value_xgb - start_value_xgb

percent_gain_xgb = (net_gain_xgb / start_value_xgb) * 100



start_value_mlp = df1['VALUE START MLP [$]'].sum()

final_value_mlp = df1['VALUE END MLP [$]'].sum()

net_gain_mlp = final_value_mlp - start_value_mlp

percent_gain_mlp = (net_gain_mlp / start_value_mlp) * 100



percent_gain_sp500 = get_price_var('^GSPC') # get percent gain of S&P500 index

percent_gain_dj = get_price_var('^DJI') # get percent gain of DOW JONES index

percent_gain_sector = PVAR['2019 PRICE VAR [%]'].mean()



MODELS_COMPARISON = pd.DataFrame([start_value_svm, final_value_svm, net_gain_svm, percent_gain_svm],

                    index=['INITIAL COST [USD]', 'FINAL VALUE [USD]', '[USD] GAIN/LOSS', 'ROI'], columns=['SVM'])

MODELS_COMPARISON['RF'] = [start_value_rf, final_value_rf, net_gain_rf, percent_gain_rf]

MODELS_COMPARISON['XGB'] = [start_value_xgb, final_value_xgb, net_gain_xgb, percent_gain_xgb]

MODELS_COMPARISON['MLP'] = [start_value_mlp, final_value_mlp, net_gain_mlp, percent_gain_mlp]

MODELS_COMPARISON['S&P 500'] = ['', '', '', percent_gain_sp500]

MODELS_COMPARISON['DOW JONES'] = ['', '', '', percent_gain_dj]

MODELS_COMPARISON['TECH SECTOR'] = ['', '', '', percent_gain_sector]



# Show the dataframe

MODELS_COMPARISON
print(53 * '=')

print(15 * ' ' + 'SUPPORT VECTOR MACHINE')

print(53 * '-')

print(classification_report(y_test, clf1.predict(X_test), target_names=['IGNORE', 'BUY']))

print(53 * '-')

print(53 * '=')

print(20 * ' ' + 'RANDOM FOREST')

print(53 * '-')

print(classification_report(y_test, clf2.predict(X_test), target_names=['IGNORE', 'BUY']))

print(53 * '-')

print(53 * '=')

print(14 * ' ' + 'EXTREME GRADIENT BOOSTING')

print(53 * '-')

print(classification_report(y_test, clf3.predict(X_test), target_names=['IGNORE', 'BUY']))

print(53 * '-')

print(53 * '=')

print(15 * ' ' + 'MULTI-LAYER PERCEPTRON')

print(53 * '-')

print(classification_report(y_test, clf4.predict(X_test), target_names=['IGNORE', 'BUY']))

print(53 * '-')