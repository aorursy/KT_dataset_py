# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



from fbprophet import Prophet

from fbprophet.plot import plot_plotly

import plotly.offline as py

py.init_notebook_mode()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

pd.set_option('display.max_columns', None)
train_path = "/kaggle/input/bits-f464-l1/train.csv"

test_path = "/kaggle/input/bits-f464-l1/test.csv"

submission_path = "/kaggle/input/bits-f464-l1/sampleSubmission.csv"



df_train = pd.read_csv(train_path)

df_test = pd.read_csv(test_path)

df_sub = pd.read_csv(submission_path)



df_train.head()
df_train.describe()
df_train.plot(x='id', y='label', figsize=(24,8))
arr = np.sort(df_train['label'].unique())
print(arr)
from statsmodels.tsa.seasonal import seasonal_decompose



series = df_train[['label']].values

result = seasonal_decompose(series, model='additive', period=10073)

result.plot()

plt.show()
df_train['season'] = (df_train.time) % 1439
all_data = pd.concat((df_train.iloc[:,1:-2],

                      df_test.iloc[:,1:]))
from scipy.stats import skew

from scipy.stats.stats import pearsonr



all_data = pd.concat((df_train.iloc[:,1:-2],

                      df_test.iloc[:,1:]))



numeric_feats = all_data.dtypes[all_data.dtypes != "int64"].index



skewed_feats = df_train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index



all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
X_train = all_data[:df_train.shape[0]]

X_train['season'] = df_train['season']

X_test = all_data[df_train.shape[0]:]

y_train = df_train.label



X_train_new = X_train.drop(['b10', 'b12', 'b26', 'b61', 'b81'], axis=1) #These features have only one value so useless

X_test_new = X_test.drop(['b10', 'b12', 'b26', 'b61', 'b81'], axis=1) 



X_test_new['season'] = df_test.time % 1439
X_train_new.head()
X_test_new.head()
num = X_train_new.columns[:-9]

num
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler

from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor



train_X = X_train_new[df_train['time'] < 18419] #80% of df_train['time'].unique()

val_X = X_train_new[df_train['time'] >= 18419]

train_y = y_train[df_train['time'] < 18419]

val_y = y_train[df_train['time'] >= 18419]



scaler1 = StandardScaler()

train_X.loc[:,num] = scaler1.fit_transform(train_X.loc[:,num])

val_X.loc[:,num] = scaler1.transform(val_X.loc[:,num])
clf = HistGradientBoostingRegressor(max_iter = 1000, random_state=0)

clf.fit(train_X.loc[:,:], train_y)

y_pred = clf.predict(val_X.loc[:,:])



error = np.sqrt(mean_squared_error(val_y,y_pred))

print('\nTimestep %d - Error %.5f' % (16116, error))
clf = RandomForestRegressor(random_state=0, n_estimators=100, verbose=1, n_jobs=-1)

clf.fit(train_X.loc[:,:], train_y)

y_pred = clf.predict(val_X.loc[:,:])



error = np.sqrt(mean_squared_error(val_y,y_pred))

print('\nTimestep %d - Error %.5f' % (16116, error))
from sklearn.ensemble import BaggingRegressor



clf = BaggingRegressor(base_estimator=HistGradientBoostingRegressor(max_iter = 1000, random_state=0),n_estimators=10, random_state=0, n_jobs=-1).fit(train_X, train_y)

y_pred = clf.predict(val_X.loc[:,:])



error = np.sqrt(mean_squared_error(val_y,y_pred))

print('\nTimestep %d - Error %.5f' % (16116, error))
test_data = X_test_new.copy()

test_data[num] = scaler1.transform(X_test_new.loc[:,num])



y_test_pred = clf.predict(test_data)
from IPython.display import FileLink



df_sub['label'] = y_test_pred

df_sub.head()

df_sub.to_csv('submission_3.csv', index=False)

FileLink('submission_3.csv')
from sklearn.ensemble import VotingRegressor



reg1 = HistGradientBoostingRegressor(random_state=0, max_iter=1000)

reg2 = RandomForestRegressor(random_state=0, n_estimators=100)

ereg = VotingRegressor([('gb', reg1), ('rf', reg2)], weights=[0.6, 0.4], n_jobs=-1)

ereg.fit(train_X, train_y)



y_pred = ereg.predict(val_X.loc[:,:])



error = np.sqrt(mean_squared_error(val_y,y_pred))

print('\nTimestep %d - Error %.5f' % (16116, error))
from sklearn.linear_model import TheilSenRegressor



regr = TheilSenRegressor(random_state=0)

regr.fit(train_X, train_y)



y_pred = regr.predict(val_X.loc[:,:])



error = np.sqrt(mean_squared_error(val_y,y_pred))

print('\nTimestep %d - Error %.5f' % (16116, error))