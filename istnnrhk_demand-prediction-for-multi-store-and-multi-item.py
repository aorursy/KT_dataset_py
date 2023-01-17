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
!echo "Quick viewing of given raw data"
!echo "## train.csv ## " ; head ../input/train.csv; echo "..." ; tail ../input/train.csv ; wc -l ../input/train.csv ; echo
!echo "## test.csv ##"   ; head ../input/test.csv ; echo "..." ; tail ../input/test.csv ; wc -l ../input/test.csv  ; echo
!echo "## sample_submission.csv ##";head ../input/sample_submission.csv;echo "..." ; tail ../input/sample_submission.csv ;wc -l ../input/sample_submission.csv
# import related libraries

# dates
from pandas import datetime

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns # advanced vizs
%matplotlib inline

# statistics
from statsmodels.distributions.empirical_distribution import ECDF

# time series analysis
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# prophet by Facebook
from fbprophet import Prophet
# Import data
train_data_csv = "../input/train.csv"
test_data_csv = "../input/test.csv"
sample_submission_csv = "../input/sample_submission.csv"

train = pd.read_csv(train_data_csv, parse_dates = True,
                    low_memory = False, index_col = 'date')
test = pd.read_csv(test_data_csv, parse_dates = True,
                   low_memory = False, index_col = 'date')
submission = pd.read_csv(sample_submission_csv)
print("Check imported data")
print()
print("In total:")
print("train.shape {} ".format(train.shape))
print("test.shape {} ".format(test.shape))
print("submission.shape {} ".format(submission.shape))
print()
print("train.columns {} ".format(train.columns))
print("test.colmuns {} ".format(test.columns))
print("submission.colmuns {} ".format(submission.columns))
print()
print("train.index {} ".format(train.index))
print("test.index {} ".format(test.index))
print("submission.index {} ".format(submission.index))


pd.set_option("display.max_rows", 20)
train.head(500)
test.head(500)
submission.head(500)
# rows which contains NA column
train[train.isna().any(axis=1)]
# rows which contains NA column
test[test.isna().any(axis=1)]
# rows which contains NA column
submission[submission.isna().any(axis=1)]
# describe - note, store and item are factor
train.describe()
# describe - note, store and item are factor
test.describe()
# describe - note, this submission data is sample
submission.describe()
pd.set_option("display.precision", 1)
# Pivot
pd.pivot_table(train, index='item', columns='store', aggfunc='count')
pd.pivot_table(train, index='item', columns='store', aggfunc='min')
pd.pivot_table(train, index='item', columns='store', aggfunc='max')
pd.pivot_table(train, index='item', columns='store', aggfunc='median')
sns.set(style = "ticks")# to format into seaborn 
c = '#386B7F' # basic color for plots
plt.figure(figsize = (12, 13))

plt.subplot(311)
cdf = ECDF(train['store'])
plt.plot(cdf.x, cdf.y, label = "statmodels", color = c);
plt.xlabel('store'); plt.ylabel('ECDF');

plt.subplot(312)
cdf = ECDF(train['item'])
plt.plot(cdf.x, cdf.y, label = "statmodels", color = c);
plt.xlabel('item'); plt.ylabel('ECDF');

plt.subplot(313)
cdf = ECDF(train['sales'])
plt.plot(cdf.x, cdf.y, label = "statmodels", color = c);
plt.xlabel('sales'); plt.ylabel('ECDF');


train['store'].hist()
train['item'].hist()
train['sales'].hist()
# check small sales values
train[train['sales'] < 2]
# data extraction
train['Year'] = train.index.year
train['Month'] = train.index.month
train['Day'] = train.index.day
train['WeekOfYear'] = train.index.weekofyear
train['DayOfYear'] = train.index.dayofweek
train['is_month_start'] = train.index.is_month_start
train['is_month_end'] = train.index.is_month_end
train['is_month_end'] = train.index.is_month_end
train['days_from_epoch'] = (train.index - pd.Timestamp("1970-01-01")).days
# sales trends
sns.catplot(data = train, x = 'Year', y = "sales", kind='point')
# sales trends
# sns.factorplot(data = train, x = 'Month', y = "sales")
sns.catplot(data = train, x = 'Month', y = "sales", kind='point')
# sales trends
# sns.factorplot(data = train, x = 'Day', y = "sales")
sns.catplot(data = train, x = 'Day', y = "sales", kind='point')
# sales trends, for each store
sns.catplot(data = train, x = 'Year', y = "sales", col='store', kind='point')
# sales trends, for each item
sns.catplot(data = train, x = 'Year', y = "sales", row='item', kind='point')
# sales trends, for each store x item
sns.catplot(data = train, x = 'Year', y = "sales",
            row = 'item', col='store', kind='point')
# timeseries plot
def tsplot(tsdf, title):
    from scipy import signal
    t = tsdf.index
    y = tsdf['sales']
    yd = signal.detrend(y)
    plt.figure(figsize=(4,3))
    plt.plot(t, y, label="Original Data")
    plt.plot(t, y-yd, "--r", label="Trend")
    plt.axis("tight")
    plt.legend(loc=0)
    plt.title(title)
    plt.show()
    return
for s in train['store'].unique():
    tmpdf = train[train['store']==s]
    # for i in tmpdf['item'].unique():
    for i in range(1,3):
        tmp2df = tmpdf[tmpdf['item']==i]
        tsplot(tmp2df, "store ID {} and item ID {}".format(s,i))
train.columns
train_X = train.copy(deep=True)
del train_X['sales']
train_y = train['sales']
# data extraction
test['Year'] = test.index.year
test['Month'] = test.index.month
test['Day'] = test.index.day
test['WeekOfYear'] = test.index.weekofyear
test['DayOfYear'] = test.index.dayofweek
test['is_month_start'] = test.index.is_month_start
test['is_month_end'] = test.index.is_month_end
test['is_month_end'] = test.index.is_month_end
test['days_from_epoch'] = (test.index - pd.Timestamp("1970-01-01")).days
test_X = test.copy(deep=True)
del test_X['id']
test_X.columns
from sklearn.ensemble import GradientBoostingRegressor

clf = GradientBoostingRegressor(n_estimators=100, learning_rate=0.25,
        max_depth=1).fit(train_X, train_y)
pred_y = clf.predict(test_X)
print ("Predict ",pred_y)
# Write submission file
out_df = pd.DataFrame({'id': test['id'].astype(np.int32), 'sales': pred_y})
out_df.to_csv('submission.csv', index=False)