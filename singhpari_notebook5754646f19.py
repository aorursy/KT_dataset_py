import numpy as np

import pandas as pd

from sklearn import preprocessing, linear_model, model_selection

from sklearn.metrics import r2_score

import gc; gc.enable()

import random

from sklearn.ensemble import RandomForestRegressor

import time

np.random.seed(888)
train = pd.read_csv("../input/train.csv",parse_dates=['date'])

test = pd.read_csv("../input/test.csv",parse_dates=['date'])

stores = pd.read_csv("../input/stores.csv")

items = pd.read_csv("../input/items.csv")

oil = pd.read_csv("../input/oil.csv", parse_dates=['date'], low_memory=False)

holiday = pd.read_csv("../input/holidays_events.csv", parse_dates=['date'])

trans = pd.read_csv("../input/transactions.csv", parse_dates=['date'])
start_time = time.time()

tcurrent   = start_time

train = train[(train['date'].dt.month == 8) & (train['date'].dt.day > 15)]

target = train['unit_sales'].values

target[target < 0.] = 0.

train['unit_sales'] = np.log1p(target)
def df_lbl_enc(df):

    for c in df.columns:

        if df[c].dtype == 'object':

            lbl = preprocessing.LabelEncoder()

            df[c] = lbl.fit_transform(df[c])

            print(c)

    return df



def df_transform(df):

    df['date'] = pd.to_datetime(df['date'])

    df['year'] = df['date'].dt.year

    df['mon'] = df['date'].dt.month

    df['day'] = df['date'].dt.day

    df['date'] = df['date'].dt.dayofweek

    df['onpromotion'] = df['onpromotion'].map({'False': 0, 'True': 1})

    df['perishable'] = df['perishable'].map({0:1.0, 1:1.25})

    df = df.fillna(-1)

    return df
items = df_lbl_enc(items)

train = pd.merge(train, items, how='left', on=['item_nbr'])

test = pd.merge(test, items, how='left', on=['item_nbr'])

del items; gc.collect();
trans = pd.read_csv("../input/transactions.csv", parse_dates=['date'])

train = pd.merge(train, trans, how='left', on=['date','store_nbr'])

test = pd.merge(test, trans, how='left', on=['date','store_nbr'])

del trans; gc.collect();

target = train['transactions'].values

target[target < 0.] = 0.

train['transactions'] = np.log1p(target)
stores = df_lbl_enc(stores)

train = pd.merge(train, stores, how='left', on=['store_nbr'])

test = pd.merge(test, stores, how='left', on=['store_nbr'])

del stores; gc.collect();



holiday = holiday[holiday['locale'] == 'National'][['date','transferred']]

holiday['transferred'] = holiday['transferred'].map({'False': 0, 'True': 1})

train = pd.merge(train, holiday, how='left', on=['date'])

test = pd.merge(test, holiday, how='left', on=['date'])

del holiday; gc.collect();



train = pd.merge(train, oil, how='left', on=['date'])

test = pd.merge(test, oil, how='left', on=['date'])

del oil; gc.collect();
train = df_transform(train)

test = df_transform(test)

col = [c for c in train if c not in ['id', 'unit_sales','perishable','transactions']]



x1, x2 = model_selection.train_test_split(train, test_size = .3, random_state = 12)





y1 = x1['unit_sales'].values

y2 = x2['unit_sales'].values

y_train = train['unit_sales'].values

def r2_func(y, pred):

    return r2_score(y, pred)
r1 = linear_model.LinearRegression(n_jobs=-1)

#r = RandomForestRegressor(n_estimators=90,max_depth =3, n_jobs=-1, random_state=2, verbose=0, warm_start=True)

r1.fit(x1[col], y1)

a1 = r2_func(y1, r1.predict(x1[col]))

a = r2_func(y2, r1.predict(x2[col]))

print('Performance: R2 = ',a1, a)
test['unit_sales'] = r1.predict(test[col])

test['unit_sales'] = test['unit_sales'].clip(lower=0.+1e-12 )

my_submission = pd.DataFrame({'id': test.id, 'unit_sales': test.unit_sales})

my_submission.to_csv('submission.csv', index=False)



from sklearn.model_selection import learning_curve

num_folds = 7



def plot_curve(ticks, train_scores, test_scores):

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)



    plt.figure()

    plt.fill_between(ticks,train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="b")

    plt.fill_between(ticks,test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="r")

    plt.plot(ticks, train_scores_mean, 'b-', label='Training score')

    plt.plot(ticks, test_scores_mean, 'r-', label='CV score')

    plt.legend()

    return plt.gca()



# Utility to plot the learning curve of a classifier for training set X and target y.

def plot_learning_curve(clf, X, y, scoring='accuracy'):

    ax = plot_curve(*learning_curve(r, X, y, cv=num_folds, scoring=scoring, train_sizes=np.linspace(0.1,1,10), n_jobs=-1))

    ax.set_title('Learning curve: {}'.format(clf.__class__.__name__))

    ax.set_xlabel('Training set size')

    ax.set_ylabel(scoring)
plot_learning_curve(r1, x1, y1)