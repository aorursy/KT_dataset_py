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

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import RFECV

from xgboost.sklearn import XGBRegressor

import sklearn.metrics

from sklearn.metrics import mean_absolute_error

import xgboost as xgb

import matplotlib.pyplot as plt

import shap
train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
train.head()
# pandas doesn't show us all the decimals

pd.options.display.precision = 15
# much better!

train.head()
# Create a training file with simple derived features



rows = 150_000

segments = int(np.floor(train.shape[0] / rows))



X_train = pd.DataFrame(index=range(segments), dtype=np.float64,

                       columns=['ave', 'std', 'max', 'min'])

y_train = pd.DataFrame(index=range(segments), dtype=np.float64,

                       columns=['time_to_failure'])



for segment in tqdm(range(segments)):

    seg = train.iloc[segment*rows:segment*rows+rows]

    x = seg['acoustic_data'].values

    y = seg['time_to_failure'].values[-1]

    

    y_train.loc[segment, 'time_to_failure'] = y

    

    X_train.loc[segment, 'ave'] = x.mean()

    X_train.loc[segment, 'std'] = x.std()

    X_train.loc[segment, 'max'] = x.max()

    X_train.loc[segment, 'min'] = x.min()
X_train.head()
scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
#choose estimator/model type for Recursive feature elimination and cross valiation

estimator = XGBRegressor()

selector = RFECV(estimator, step=1, min_features_to_select=1, cv=10, scoring='neg_mean_absolute_error')



#fit the model, get a rank of the variables, and a matrix of the selected X variables

selector = selector.fit(X_train_scaled, y_train.values.flatten())





#PLot # of features selected vs. Model Score

plt.figure()

plt.title('XGB CV score vs No of Features')

plt.xlabel("Number of features selected")

plt.ylabel("Neg Mean Squared Error")

plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)

plt.show()



#get rank of X model features

rank = selector.ranking_

#Subset features to those selected by recursive feature elimination

#X_train_scaled = X_train_scaled[:,selector.support_ ] 



y_pred = selector.predict(X_train_scaled)





#svm = NuSVR()

#svm.fit(X_train_scaled, y_train.values.flatten())

#y_pred = svm.predict(X_train_scaled)
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
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
X_test = pd.DataFrame(columns=X_train.columns, dtype=np.float64, index=submission.index)

for seg_id in X_test.index:

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    

    x = seg['acoustic_data'].values

    

    X_test.loc[seg_id, 'ave'] = x.mean()

    X_test.loc[seg_id, 'std'] = x.std()

    X_test.loc[seg_id, 'max'] = x.max()

    X_test.loc[seg_id, 'min'] = x.min()
X_test_scaled = scaler.transform(X_test)

#X_test_scaled = X_test_scaled[:,selector.support_ ] 

submission['time_to_failure'] = selector.predict(X_test_scaled)

submission.to_csv('submission.csv')