# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_submission = pd.read_csv('../input/sample_submission.csv')

df_submission.head()
from calendar import day_abbr

from datetime import datetime

from time import mktime



def prepocess_train(df, df_store, df_train):

    df = pd.concat((df.set_index('Store'), df_store.loc[df.Store]), axis='columns')

    for x in np.unique(df_train.DayOfWeek) - 1:

        df['On{}'.format(day_abbr[x])] = np.where(df.DayOfWeek == x + 1, 1, 0)

    df['Timestamp'] = df.Date.apply(lambda x: mktime(datetime.strptime(x, '%Y-%m-%d').timetuple()))

    for x in np.unique(df_train.StateHoliday):

        df['WithStateHoliday{}'.format(x.capitalize())] = np.where(df.StateHoliday == x, 1, 0)

    return df.drop(['DayOfWeek', 'Date', 'StateHoliday'], axis='columns').fillna(0)
df_store = pd.read_csv('../input/store.csv')

df_store['CompetitionOpenSinceTimestamp'] = df_store.CompetitionOpenSinceYear.fillna(1917).astype(int).astype(str) + '-' + df_store.CompetitionOpenSinceMonth.fillna(1).astype(int).astype(str)

df_store['CompetitionOpenSinceTimestamp'] = df_store.CompetitionOpenSinceTimestamp.apply(lambda x: mktime(datetime.strptime(x, '%Y-%m').timetuple()))

for value in np.unique(df_store.StoreType):

    df_store['WithStoreType{}'.format(value.capitalize())] = np.where(df_store.StoreType == value, 1, 0)

for value in np.unique(df_store.Assortment):

    df_store['WithAssortment{}'.format(value.capitalize())] = np.where(df_store.Assortment == value, 1, 0)

df_store['Promo2SinceTimestamp'] = df_store.Promo2SinceYear.fillna(1917).astype(int).astype(str) + '-' + df_store.Promo2SinceWeek.fillna(1).astype(int).astype(str)

df_store['Promo2SinceTimestamp'] = df_store.Promo2SinceTimestamp.apply(lambda x: mktime(datetime.strptime(x, '%Y-%W').timetuple()))

for value in np.unique(df_store.PromoInterval.astype(str)):

    df_store['WithPromoInterval{}'.format(value[0].capitalize())] = np.where(df_store.PromoInterval.astype(str) == value, 1, 0)

df_store = df_store.drop(['StoreType', 'Assortment', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval'], axis='columns').fillna(1).set_index('Store')

df_store.head()
df_train = pd.read_csv('../input/train.csv', low_memory=False)

df_test = pd.read_csv('../input/test.csv', low_memory=False)

df_test = prepocess_train(df_test, df_store, df_train).set_index('Id')

df_train = prepocess_train(df_train, df_store, df_train)

df_train.head()
df_test.head()
X_train = df_train.drop(['Sales', 'Customers'], axis='columns')

y_train = df_train.Sales

X_train.head()
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split



def select_model(x, y):

    best_accuracy = 0

    best_model = None

    losses = []

    model = RandomForestRegressor(min_samples_split=25, min_samples_leaf=5, n_jobs=-1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    for model.max_features in range(2, x.shape[1]):

        model.fit(x_train, y_train)

        losses.append(mean_absolute_error(y_test, model.predict(x_test)))

        if best_model is None or losses[-1] < best_accuracy:

            best_accuracy = losses[-1]

            best_model = model

    plt.plot(range(2, x.shape[1]), losses)

    plt.legend(['loss'])

    plt.title('Mean absolute error')

    plt.xlabel('C')

    plt.xscale('log')

    plt.ylabel('MAE')

    return best_model
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split



def select_model(x, y):

    best_accuracy = 0

    best_model = RandomForestRegressor(min_samples_split=25, min_samples_leaf=5, n_jobs=-1)

    losses = []

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    for n_features in range(2, x.shape[1]):

        model = best_model

        model.max_features = n_features

        model.fit(x_train, y_train)

        losses.append(mean_absolute_error(y_test, model.predict(x_test)))

        if best_model is None or losses[-1] < best_accuracy:

            best_accuracy = losses[-1]

            best_model.max_features = n_features

    plt.plot(range(2, x.shape[1]), losses)

    plt.legend(['loss'])

    plt.title('Mean absolute error')

    plt.xlabel('n_features')

    plt.ylabel('MAE')

    return best_model
regr = select_model(X_train, y_train)
regr.n_estimators = 100

regr.fit(X_train, y_train)
df_submission.Sales = np.round(regr.predict(df_test))

df_submission.to_csv('submission.csv', index=False)