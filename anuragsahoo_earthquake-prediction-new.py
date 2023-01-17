# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.stats import skew,kurtosis

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.linear_model import LinearRegression

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from scipy.stats import kurtosis

a=np.array([1,5,8,6,4,2])

kurtosis(a)
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler

from sklearn.svm import NuSVR

from sklearn.metrics import mean_absolute_error

from catboost import CatBoostRegressor
train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
train.head(10)
# pandas doesn't show us all the decimals

pd.options.display.precision = 15
# much better!

train.head()

#train.shape
# Create a training file with simple derived features

from scipy.stats import kurtosis,skew

rows = 150_000

segments = int(np.floor(train.shape[0] / rows))



X_train = pd.DataFrame(index=range(segments), dtype=np.float64,

         columns=['ave', 'std', 'max', 'min','kurt','skew','25per','50per','75per'])

#X_train=pd.DataFrame(columns=['ave','std','max','min'])

y_train = pd.DataFrame(index=range(segments), dtype=np.float64,

                       columns=['time_to_failure'])

#y_train=pd.DataFrame(columns=['time_to_failure'])

for segment in tqdm(range(segments)):

    seg = train.iloc[segment*rows:segment*rows+rows]

    x = seg['acoustic_data'].values

    y = seg['time_to_failure'].values[-1]

    #print(seg['acoustic_data'].values)

    #print(seg['time_to_failure'].values[-1])

    y_train.loc[segment, 'time_to_failure'] = y

    

    X_train.loc[segment, 'ave'] = x.mean()

    X_train.loc[segment, 'std'] = x.std()

    X_train.loc[segment, 'max'] = x.max()

    X_train.loc[segment, 'min'] = x.min()

    #X_train.loc[segment, 'mad'] = x.mad()

    X_train.loc[segment, 'kurt'] = kurtosis(x)

    X_train.loc[segment, 'skew'] = skew(x)

    X_train.loc[segment, '25per'] = np.quantile(x,0.25)

    X_train.loc[segment, '50per'] = np.quantile(x,0.50)

    X_train.loc[segment, '75per'] = np.quantile(x,0.75)
print(X_train.head())

print(y_train.head())

print(X_train.shape[0],y_train.shape[0])
scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)

print(X_train_scaled)
svm = NuSVR()

svm.fit(X_train_scaled, y_train.values.flatten())

y_pred = svm.predict(X_train_scaled)
plt.figure(figsize=(6, 6))

plt.scatter(y_train.values.flatten(), y_pred)

plt.xlim(0, 20)

plt.ylim(0, 20)

plt.xlabel('actual', fontsize=12)

plt.ylabel('predicted', fontsize=12)

plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])

plt.show()
score = mean_absolute_error(y_train.values.flatten(), y_pred)

print(f'Score: {score:0.3f}')
linearRegressor = LinearRegression()

linearRegressor.fit(X_train_scaled, y_train.values.flatten())

y_pred1 = linearRegressor.predict(X_train_scaled)

plt.figure(figsize=(6, 6))

plt.scatter(y_train.values.flatten(), y_pred1)

plt.xlim(0, 20)

plt.ylim(0, 20)

plt.xlabel('actual', fontsize=12)

plt.ylabel('predicted', fontsize=12)

plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])

plt.show()
score = mean_absolute_error(y_train.values.flatten(), y_pred1)

print(f'Score: {score:0.3f}')
catboostreg=CatBoostRegressor(iterations=10000,loss_function='MAE',boosting_type='Ordered')

catboostreg.fit(X_train,y_train,silent=True)

print(catboostreg.best_score_)
#no impact on training with Cat Boost Regressor on scaled data

catboostreg1=CatBoostRegressor(iterations=10000,loss_function='MAE',boosting_type='Ordered')

catboostreg1.fit(X_train_scaled,y_train,silent=True)

print(catboostreg1.best_score_)
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
X_test = pd.DataFrame(columns=X_train.columns, dtype=np.float64, index=submission.index)

print(X_test.index)

for seg_id in X_test.index:

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    

    x = seg['acoustic_data'].values

    #print(seg_id)

    #print(x.mean())

    X_test.loc[seg_id, 'ave'] = x.mean()

    X_test.loc[seg_id, 'std'] = x.std()

    X_test.loc[seg_id, 'max'] = x.max()

    X_test.loc[seg_id, 'min'] = x.min()

    X_test.loc[seg_id, 'kurt'] = kurtosis(x)

    X_test.loc[seg_id, 'skew'] = skew(x)

    X_test.loc[seg_id, '25per'] = np.quantile(x,0.25)

    X_test.loc[seg_id, '50per'] = np.quantile(x,0.50)

    X_test.loc[seg_id, '75per'] = np.quantile(x,0.75)
X_test
X_test_scaled = scaler.transform(X_test)

submission['time_to_failure'] = catboostreg.predict(X_test_scaled)

#submission['time_to_failure'] = svm.predict(X_test_scaled)

submission.to_csv('submission_cbg.csv')

print("Done")