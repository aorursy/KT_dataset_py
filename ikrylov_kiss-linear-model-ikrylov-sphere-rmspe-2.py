import numpy as np

import pandas as pd

from sklearn.linear_model import LinearRegression
test = pd.read_csv("../input/test.csv",parse_dates=[3],index_col="Id",dtype={"StateHoliday":np.str})

train = pd.read_csv("../input/train.csv",parse_dates=[2],dtype={"StateHoliday":np.str})

print(test.dtypes) # Kaggle doesn't seem to support Python 2.7. Oh well.

print(train.dtypes)
set(test.Store.values).issubset(train.Store.values)

# if true, I can predict sales per store, not the whole model with store info joined
train = train.loc[train.Sales > 0]

# Any day and store with 0 sales is ignored in scoring => no reason to predict them, either
def prepare(df):

    # transform the date into something human-meaningful

    df['Year'] = pd.DatetimeIndex(df.Date).year

    df['Month'] = pd.DatetimeIndex(df.Date).month

    df['Day'] = pd.DatetimeIndex(df.Date).day

    # encode StateHolidays into numbers

    # Since there are only 'a' state holidays in test set, I can probably map a, b, c into 1

    df[df.StateHoliday != '0'] = 1

    df.StateHoliday = pd.to_numeric(df.StateHoliday)

    return df
train = prepare(train);

test = prepare(test);

print(train.dtypes)

print(test.dtypes)
train.head()
test.head()
# Curses! NA! Foiled again!

test.iloc[pd.isnull(test).any(1).nonzero()]
test_nona = test.dropna().copy()

stores = set(test_nona.Store.values)

test_nona['Sales'] = 0 # create a column to be filled
columns = ['DayOfWeek','Open','Promo','SchoolHoliday','Year','Month','Day','StateHoliday']

# Customers are not present in test, not worth using

for store in stores: # takes *FOREVER* to run

    # pandas throws "IndexingError: Unalignable boolean Series key provided" if I index df directly

    # well, it must think it's so clever

    train_store_indices = (train.Store.values==store)

    

    y_train = train.Sales.values[train_store_indices]

    X_train = train[columns].values[train_store_indices]

    

    model = LinearRegression(normalize=True,n_jobs=-1).fit(X_train,y_train)

    

    test_store_indices = (test_nona.Store==store)

    X_test = test_nona[columns].values[test_store_indices]

    

    test_nona.Sales.values[test_store_indices] = model.predict(X_test)
test['Sales'] = test_nona.Sales

test = test.fillna(0) # we didn't predict some stores with NAs, tell Kaggle to ignore them
test[['Sales']].to_csv("submission.csv")