# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns



sns.set(style='whitegrid', palette='muted')



data_train = pd.read_csv('../input/train.csv',

                         dtype={

                             'StateHoliday': 'category',

                             'SchoolHoliday': 'category'},

                         parse_dates=['Date'])

data_test = pd.read_csv('../input/test.csv',

                        dtype={

                            'StateHoliday': 'category',

                            'SchoolHoliday': 'category'},

                        parse_dates=['Date'])

data_store = pd.read_csv('../input/store.csv',

                         dtype={

                             'StoreType': 'category',

                             'Assortment': 'category',

                             'CompetitionOpenSinceMonth': float,

                             'CompetitionOpenSinceYear': float,

                             'Promo2': float,

                             'Promo2SinceWeek': float,

                             'Promo2SinceYear': float})

data_train = pd.merge(data_train, data_store, on='Store', how='left')

data_train.head(5)
print('Total number of samples:', data_train.shape[0])

print('')

data_train.info()

print('')

print(data_train.iloc[:, 1:].describe())
counts = data_train.isnull().sum()

print('Missing value counts:')

print(counts)

plt.figure(figsize=(10, 3))

g = sns.barplot(counts.index, counts.values)

g.set_xticklabels(g.get_xticklabels(), rotation=90)

plt.show()
fig, axes = plt.subplots(2, 2, figsize=(15, 8))

sns.distplot(data_train['Sales'].dropna(), bins=20, ax=axes[0][0]);

sns.distplot(data_train['CompetitionDistance'].dropna(), bins=20, ax=axes[0][1]);

sns.boxplot(x=data_train['Sales'].dropna(), ax=axes[1][0])

sns.boxplot(x=data_train['CompetitionDistance'].dropna(), ax=axes[1][1])

plt.show()
plt.figure(figsize=(15, 5))

data_train_sales = data_train[['Date', 'Sales']]

data_train_sales_1 = data_train_sales.groupby(pd.Grouper(key='Date', freq='7D')).mean()

ax = sns.lineplot(x=data_train_sales_1.index, y=data_train_sales_1['Sales'])

ax.set_title('Average daily sales per 7 days from 2013-01 to 2015-07')

plt.show()
plt.figure(figsize=(15, 5))

data_train_customers = data_train[['Date', 'Customers']]

data_train_customers_1 = data_train_customers.groupby(pd.Grouper(key='Date', freq='7D')).mean()

ax = sns.lineplot(x=data_train_customers_1.index, y=data_train_customers_1['Customers'])

ax.set_title('Average daily customers per 7 days from 2013-01 to 2015-07')

plt.show()
plt.figure(figsize=(15, 5))

sns.scatterplot(x=data_train_sales_1['Sales'], y=data_train_customers_1['Customers']).set_title('Sales and Customers')

plt.show()
data_train_open_sl_cstm = data_train[['Sales', 'Customers']]

data_train_open_sl_cstm.corr()
plt.figure(figsize=(15, 4))

data_train_wd = data_train.copy()

data_train_wd['Year'] = data_train_wd['Date'].dt.strftime('%Y')

data_train_wd = data_train_wd.groupby(['Year', 'DayOfWeek']).mean().reset_index()

sns.barplot(x='DayOfWeek', y='Sales', hue='Year', palette='pastel', data=data_train_wd)

data_train_wd = data_train_wd.groupby(['DayOfWeek']).mean().reset_index()

ax = sns.lineplot(x=data_train_wd.index, y=data_train_wd['Sales'], color='#c64d4f')

ax.set_title('Average daily sales by day of week')

ax.legend_.set_title('Year')

plt.show()
plt.figure(figsize=(15, 4))

data_train_m = data_train.copy()

data_train_m['Year'] = data_train_m['Date'].dt.strftime('%Y')

data_train_m['Month'] = data_train_m['Date'].dt.strftime('%m')

data_train_m = data_train_m.groupby(['Year', 'Month']).mean().reset_index()

sns.barplot(x='Month', y='Sales', hue='Year', palette='pastel', data=data_train_m)

data_train_m = data_train_m.groupby(['Month']).mean().reset_index()

ax = sns.lineplot(x=data_train_m.index, y=data_train_m['Sales'], color='#c64d4f')

ax.set_title('Average daily sales by month')

ax.legend_.set_title('Year')

plt.show()
plt.figure(figsize=(15, 4))

data_train_ = data_train.copy()

data_train_ = data_train_[['Sales', 'Date', 'Promo', 'Promo2']]

data_train_['Promo_Promo2'] = data_train_['Promo'] & data_train_['Promo2']

data_train_ = data_train_.groupby([pd.Grouper(key='Date', freq='30D'), 'Promo', 'Promo_Promo2']).mean().reset_index()

ax = sns.lineplot(x='Date', y='Sales', hue='Promo', style='Promo_Promo2', data=data_train_, markers=True, dashes=False)

ax.set_title('Average daily sales per 30 days by promo')

plt.show()
plt.figure(figsize=(10, 3))

ax = sns.boxplot(x='StoreType', y='Sales', hue='Assortment', palette='pastel', data=data_train)

ax.set_title('Sales by StoreType and Assortment')

plt.show()
# Create utilities for preprocessing

def read_csv(files):

    data_train = pd.read_csv(files[0],

                             dtype={

                                 'StateHoliday': 'category',

                                 'SchoolHoliday': 'int'},

                             parse_dates=['Date'])



    data_test = pd.read_csv(files[1],

                            dtype={

                                'StateHoliday': 'category',

                                'SchoolHoliday': 'int'},

                            parse_dates=['Date'])



    data_store = pd.read_csv(files[2],

                             dtype={

                                 'StoreType': 'category',

                                 'Assortment': 'category',

                                 'CompetitionOpenSinceMonth': float,

                                 'CompetitionOpenSinceYear': float,

                                 'Promo2': float,

                                 'Promo2SinceWeek': float,

                                 'Promo2SinceYear': float,

                                 'PromoInterval': str})



    return data_train, data_test, data_store



def combine(data, data_store):

    return pd.merge(data, data_store, on='Store', how='left')



def preprocess(data_train, data_test):

    data_train_ = data_train.copy()

    data_test_ = data_test.copy()



    data_train_ = data_train_[(data_train_['Sales'] > 0) & (data_train_['Open'] != 0)]

    data_test_['Open'] = data_test_['Open'].fillna(1)



    def process(data):

        data['Year'] = data['Date'].dt.year

        data['Month'] = data['Date'].dt.month

        data['Day'] = data['Date'].dt.day

        data['DayOfWeek'] = data['Date'].dt.dayofweek

        data['DayOfYear'] = data['Date'].dt.dayofyear

        data['WeekOfYear'] = data['Date'].dt.weekofyear

        data['Quarter'] = (data['Date'].dt.month - 1) // 3 + 1



        mappings = {'0': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4}

        data['StoreType'].replace(mappings, inplace=True)

        data['Assortment'].replace(mappings, inplace=True)

        data['StateHoliday'].replace(mappings, inplace=True)



        data['CompetitionDistance'] = data['CompetitionDistance'].fillna(data['CompetitionDistance'].median())



        # Extend features

        data['CompetitionOpen'] = 12 * (data['Year'] - data['CompetitionOpenSinceYear']) + (data['Month'] - data['CompetitionOpenSinceMonth']).apply(lambda x: x if x > 0 else 0)



        data['Promo2Open'] = 12 * (data['Year'] - data['Promo2SinceYear']) + (data['WeekOfYear'] - data['Promo2SinceWeek']) / 4.0

        data['Promo2Open'] = data['Promo2Open'].apply(lambda x: x if x > 0 else 0)



        data['PromoInterval'] = data['PromoInterval'].fillna('')

        data['InPromoMonth'] = data.apply(lambda x: 1 if (x['Date'].strftime('%b') if not x['Date'].strftime('%b') == 'Sep' else 'Sept') in x['PromoInterval'].split(',') else 0, axis=1)



        data_meanlog_salesbystore = data_train_.groupby(['Store'])['Sales'].mean().reset_index(name='MeanLogSalesByStore')

        data_meanlog_salesbystore['MeanLogSalesByStore'] = np.log1p(data_meanlog_salesbystore['MeanLogSalesByStore'])

        data = data.merge(data_meanlog_salesbystore, on=['Store'], how='left', validate='m:1')



        data_meanlog_salesbydow = data_train_.groupby(['DayOfWeek'])['Sales'].mean().reset_index(name='MeanLogSalesByDOW')

        data_meanlog_salesbydow['MeanLogSalesByDOW'] = np.log1p(data_meanlog_salesbystore['MeanLogSalesByStore'])

        data = data.merge(data_meanlog_salesbydow, on=['DayOfWeek'], how='left', validate='m:1')



        data_meanlog_salesbymonth = data_train_.groupby(['Month'])['Sales'].mean().reset_index(name='MeanLogSalesByMonth')

        data_meanlog_salesbymonth['MeanLogSalesByMonth'] = np.log1p(data_meanlog_salesbymonth['MeanLogSalesByMonth'])

        data = data.merge(data_meanlog_salesbymonth, on=['Month'], how='left', validate='m:1')



        return data



    features = [

        'Store', 'DayOfWeek', 'Year', 'Month', 'Day', 'DayOfYear', 'WeekOfYear', 'Quarter', 'Open', 'Promo',

        'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',

        'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'CompetitionOpen',

        'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'Promo2Open', 'InPromoMonth',

        'MeanLogSalesByStore', 'MeanLogSalesByDOW', 'MeanLogSalesByMonth']



    data_train_ = process(data_train_)

    data_test_ = process(data_test_)



    return (data_train_[features], np.log1p(data_train_['Sales'])), data_test_[features]
pd.set_option('display.max_columns', None)  

pd.set_option('display.expand_frame_repr', False)

pd.set_option('max_colwidth', -1)



data_train, data_test, data_store = read_csv(['../input/train.csv', '../input/test.csv', '../input/store.csv'])

data_train = combine(data_train, data_store)

data_test = combine(data_test, data_store)

(X_train, y_train), X_test = preprocess(data_train, data_test)
# Evaluation function

def rmspe(y_true, y_pred):

    err = np.sqrt(np.mean((1 - y_pred / y_true) ** 2))

    return err



# Evaluation function adapted to XGBoost

def rmspe_xgb(y_pred, y_true):

    y_true = y_true.get_label()

    err = rmspe(y_true, y_pred)

    return 'rmspe', err
import xgboost as xgb



class Model:

    def __init__(self, params=None, **kwargs):

        self.params = params

        self.kwargs = kwargs



    def train(self, X, y):

        X_train, X_test, y_train, y_test = X[41088:], X[:41088], y[41088:], y[:41088]

        dtrain = xgb.DMatrix(X_train, label=y_train)

        dvalid = xgb.DMatrix(X_test, label=y_test)

        watchlist = [(dvalid, 'eval'), (dtrain, 'train')]

        self.bst = xgb.train(self.params, dtrain, self.kwargs['num_boost_round'], evals=watchlist,

                             feval=rmspe_xgb, early_stopping_rounds=self.kwargs['early_stopping_rounds'],

                             verbose_eval=True)



    def predict(self, X, weight=0.995):

        y_pred = np.expm1(weight * self.bst.predict(xgb.DMatrix(X)))

        return y_pred



    def save_model(self, filename):

        joblib.dump(self.bst, filename)



    def load_model(self, filename):

        self.bst = joblib.load(filename)
# Train the model

params = {

    'eta': 0.03,

    'max_depth': 10,

    'subsample': 0.8,

    'colsample_bytree': 0.7,

    'lambda': 0.2,

    'silent': 1,

    'seed': 12

}



m = Model(params, num_boost_round=5000, early_stopping_rounds=50)

m.train(X_train, y_train)
def correction(model, y_train):

    y_true, y_pred = y_train[:41088], model.bst.predict(xgb.DMatrix(X_train[:41088]))

    weights = np.arange(0.98, 1.02, 0.005)

    errors = []

    

    for w in weights:

        error = rmspe(np.expm1(y_true[:41088]), np.expm1(y_pred * w))

        errors.append(error)



    plt.plot(weights, errors)

    plt.xlabel('weight')

    plt.ylabel('RMSPE')

    plt.title('RMSPE Curve')

    

    idx = errors.index(min(errors))

    print('Best weight is {}, RMSPE is {:.4f}'.format(weights[idx], min(errors)))

    

correction(m, y_train)
import os



weight = 1.0

y_pred = m.predict(X_test, weight=weight)



result = pd.DataFrame({'Id': data_test['Id'], 'Sales': y_pred})

result.to_csv('submission.csv', index=False)

result