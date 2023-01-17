import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from itertools import product
sales=pd.read_csv("../input/sales.csv")

cat=pd.read_csv("../input/categories.csv")

items=pd.read_csv("../input/items.csv")

shops=pd.read_csv("../input/shops.csv")

test=pd.read_csv("../input/test.csv")
print("sales Columns::",sales.columns,"\n","cat::",cat.columns,"items collumns::",items.columns,"\n","shops columns::",shops.columns,"\n","test collumns::",test.columns)
import datetime

sales.date = sales.date.apply(lambda x:datetime.datetime.strptime(x, "%d.%m.%Y"))

sales.info()
for i in range(4, 10):

    sales["lag_{}".format(i)] = sales.item_cnt_day.shift(i)

print("done")
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(16,8))

sns.lineplot(x=sales.date, y=sales.item_cnt_day)
plt.figure(figsize=(16,8))

sns.lineplot(x=sales.date,y=sales.item_cnt_day.rolling(window=120, center=False).mean())
X = sales.dropna().drop(["item_cnt_day", "item_price", "date_block_num"], axis=1)
X = X.iloc[:,1:]

X = X.reset_index(drop=True)

X.head()
y = sales.dropna().item_cnt_day

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

X = sc_x.fit_transform(X)

X = pd.DataFrame(X)
def TimeSeriesTrainTestSplit(x, y, test_size):

    

        test_index = int(len(X)*(1-test_size))

    

        X_train = X.iloc[:test_index]

        y_train = y.iloc[:test_index]

        X_test = X.iloc[test_index:]

        y_test = y.iloc[test_index:]

        return X_train, y_train, X_test, y_test

    

X_train, y_train, X_test, y_test = TimeSeriesTrainTestSplit(X,y, 0.2)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)

from sklearn.model_selection import TimeSeriesSplit

ts_cross_val = TimeSeriesSplit(n_splits=5)

from sklearn.model_selection import cross_val_score

cv = cross_val_score(regressor, X_train, y_train, cv = ts_cross_val, scoring= "neg_mean_squared_error")
y_pred = regressor.predict(X_test)

plt.figure(figsize=(20,8))

plt.plot(regressor.predict(X_test[-400:]), "y", label="prediction", linewidth=2.0)

plt.plot(y_test.values[-400:], "g", label="real_values", linewidth=2.0)

plt.legend(loc="best")
test = test.merge(sales, how="left", on = ["shop_id", "item_id"], copy=False)

test = test.drop_duplicates()

test.info()
test.drop(["ID", "date", "date_block_num", "item_cnt_day", "item_price"], axis=1, inplace=True)

test = test.dropna()

y_pred = regressor.predict(test)

y_pred
submission = pd.DataFrame(y_pred

        )



submission.to_csv("sample_submission.csv", index=False)

print(submission.shape)
