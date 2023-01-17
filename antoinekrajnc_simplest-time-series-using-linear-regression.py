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
sales = pd.read_csv("../input/sales_train.csv")
import datetime
sales.date = sales.date.apply(lambda x:datetime.datetime.strptime(x, "%d.%m.%Y"))
sales.info()
for i in range(4, 10):
    sales["lag_{}".format(i)] = sales.item_cnt_day.shift(i)
print("done")
sales.head()
import seaborn as sns
import matplotlib.pyplot as plt
"""plt.figure(figsize=(16,8))
sns.lineplot(x=sales.date, y=sales.item_cnt_day)"""
"""plt.figure(figsize=(16,8))
sns.lineplot(x=sales.date,y=sales.item_cnt_day.rolling(window=120, center=False).mean())"""
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
cv
y_pred = regressor.predict(X_test)
plt.figure(figsize=(20,8))
plt.plot(regressor.predict(X_test[-400:]), "y", label="prediction", linewidth=2.0)
plt.plot(y_test.values[-400:], "g", label="real_values", linewidth=2.0)
plt.legend(loc="best")
test = pd.read_csv("../input/test.csv")
test.info()
test = test.merge(sales, how="left", on = ["shop_id", "item_id"], copy=False)
test = test.drop_duplicates()
test.info()
test.drop(["ID", "date", "date_block_num", "item_cnt_day", "item_price"], axis=1, inplace=True)
test = test.dropna()
y_pred = regressor.predict(test)
y_pred
submission = pd.DataFrame(y_pred)
submission
submission = submission.dropna()
submission.rename(columns={"index":"ID", 0:"item_cnt_month"})
submission = submission.iloc[:214201,:]
submission.to_csv("submission.csv", header=True)
submission
