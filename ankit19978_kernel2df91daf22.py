# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/nifty50-stock-market-data/HDFC.csv")
df.dtypes
df = df.dropna(axis=1)
df
df = df.drop(["Symbol","Series","Prev Close"],axis=1)
df.iloc[:,0] = pd.to_datetime(df['Date'],format='%Y-%m-%d')
df.dtypes
df.columns = [str(i).lower().replace(' ', '_') for i in df.columns]
df
import matplotlib.pyplot as plt

fig= plt.figure(figsize=(10,8))
plt.plot(df["date"],df["close"])
plt.show()
test_size = 0.2                 # proportion of dataset to be used as test set
cv_size = 0.2                   # proportion of dataset to be used as cross-validation set
Nmax = 21                       # for feature at day t, we use lags from t-1, t-2, ..., t-N as features
                                # Nmax is the maximum N we are going to test
num_cv = int(cv_size*len(df))
num_test = int(test_size*len(df))
num_train = len(df)-num_cv-num_test
print(num_cv,num_test,num_train)
train = df[:num_train]
cv = df[num_train:num_train+num_cv]
train_cv = df[:num_train+num_cv]
test = df[num_train+num_cv:]
fig= plt.figure(figsize=(10,8))
plt.plot(train["date"],train["close"])
plt.plot(cv["date"],cv["close"])
plt.plot(test["date"],test["close"])
plt.show()
def get_preds_mov_avg(df, target_col, N, pred_min, offset):
    """
    Given a dataframe, get prediction at timestep t using values from t-1, t-2, ..., t-N.
    Using simple moving average.
    Inputs
        df         : dataframe with the values you want to predict. Can be of any length.
        target_col : name of the column you want to predict e.g. 'adj_close'
        N          : get prediction at timestep t using values from t-1, t-2, ..., t-N
        pred_min   : all predictions should be >= pred_min
        offset     : for df we only do predictions for df[offset:]. e.g. offset can be size of training set
    Outputs
        pred_list  : list. The predictions for target_col. np.array of length len(df)-offset.
    """
    pred_list = df[target_col].rolling(window = N, min_periods=1).mean() # len(pred_list) = len(df)
    # Add one timestep to the predictions
    pred_list = np.concatenate((np.array([np.nan]), np.array(pred_list[:-1])))
    
    # If the values are < pred_min, set it to be pred_min
    pred_list = np.array(pred_list)
    pred_list[pred_list < pred_min] = pred_min
    return pred_list[offset:]

def get_mape(y_true, y_pred): 
    """
    Compute mean absolute percentage error (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
import math
from sklearn.metrics import mean_squared_error
RMSE = []
mape = []
Nmax = 21
for N in range(1, Nmax+1): # N is no. of samples to use to predict the next value
    est_list = get_preds_mov_avg(train_cv, 'close', N, 0, num_train)
    
    cv.loc[:, 'est' + '_N' + str(N)] = est_list
    RMSE.append(math.sqrt(mean_squared_error(est_list, cv['close'])))
    mape.append(get_mape(cv['close'], est_list))
print('RMSE = ' + str(RMSE))
print('MAPE = ' + str(mape))
df.head()
fig= plt.figure(figsize=(10,8))
plt.plot(range(1,Nmax+1),RMSE)
plt.show()
fig= plt.figure(figsize=(10,8))
plt.plot(range(1,Nmax+1),mape)
plt.show()
cv
fig= plt.figure(figsize=(10,8))
plt.plot(train["date"],train["close"])
plt.plot(cv["date"],cv["close"])
plt.plot(test["date"],test["close"])
plt.plot(cv["date"],cv["est_N1"])
plt.plot(cv["date"],cv["est_N21"])
plt.show()
fig= plt.figure(figsize=(10,8))
plt.plot(cv["date"],cv["close"],label="close")
plt.plot(cv["date"],cv["est_N1"])
plt.plot(cv["date"],cv["est_N2"])
plt.show()
N_opt = 2
est_list = get_preds_mov_avg(df, 'close', N_opt, 0, num_train+num_cv)
test.loc[:, 'est' + '_N' + str(N_opt)] = est_list
print("RMSE = %0.3f" % math.sqrt(mean_squared_error(est_list, test['close'])))
print("MAPE = %0.3f%%" % get_mape(test['close'], est_list))
test.head()
fig= plt.figure(figsize=(10,8))
plt.plot(train["date"],train["close"])
plt.plot(cv["date"],cv["close"])
plt.plot(test["date"],test["close"])
plt.plot(test["date"],test["est_N2"])
plt.show()
fig= plt.figure(figsize=(10,8))
plt.plot(test["date"],test["close"])
plt.plot(test["date"],test["est_N2"])
plt.show()