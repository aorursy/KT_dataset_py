# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns



%matplotlib inline

# Any results you write to the current directory are saved as output.
TRAINPATH = os.path.join("..", "input", "train.csv")

TESTPATH = os.path.join("..", "input", "test.csv")
df = pd.read_csv(TRAINPATH, index_col=0)

df.head()
df_test = pd.read_csv(TESTPATH)

df_test.head()
df.info()
df.duplicated().sum()
## suppression des valeurs récurrentes



df = df.drop_duplicates()

df.duplicated().sum()



# valeurs nulles 



df.isna().sum()
fig, ax = plt.subplots(figsize=(12, 5))

df.boxplot(['trip_duration'], fontsize=12)

fig.suptitle('Visualisation des outliers', fontsize=20)
CAT_VARS = ['store_and_fwd_flag']
for col in CAT_VARS:

    df[col] = df[col].astype('category').cat.codes

df.head()
for col in CAT_VARS:

    df_test[col] = df_test[col].astype('category').cat.codes

df_test.head()
from sklearn.decomposition import PCA
coords = np.vstack((df[['pickup_latitude', 'pickup_longitude']].values,

                    df[['dropoff_latitude', 'dropoff_longitude']].values,

                    df_test[['pickup_latitude', 'pickup_longitude']].values,

                    df_test[['dropoff_latitude', 'dropoff_longitude']].values))



pca = PCA().fit(coords)



# Train

df['pickup_pca0'] = pca.transform(df[['pickup_latitude', 'pickup_longitude']])[:, 0]

df['pickup_pca1'] = pca.transform(df[['pickup_latitude', 'pickup_longitude']])[:, 1]

df['dropoff_pca0'] = pca.transform(df[['dropoff_latitude', 'dropoff_longitude']])[:, 0]

df['dropoff_pca1'] = pca.transform(df[['dropoff_latitude', 'dropoff_longitude']])[:, 1]



# Test

df_test['pickup_pca0'] = pca.transform(df_test[['pickup_latitude', 'pickup_longitude']])[:, 0]

df_test['pickup_pca1'] = pca.transform(df_test[['pickup_latitude', 'pickup_longitude']])[:, 1]

df_test['dropoff_pca0'] = pca.transform(df_test[['dropoff_latitude', 'dropoff_longitude']])[:, 0]

df_test['dropoff_pca1'] = pca.transform(df_test[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
df.head()
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])

df_test['pickup_datetime'] = pd.to_datetime(df_test['pickup_datetime'])
df['hour'] = df.pickup_datetime.dt.hour

df['day'] = df.pickup_datetime.dt.dayofweek

df['month'] = df.pickup_datetime.dt.month

df_test['hour'] = df_test.pickup_datetime.dt.hour

df_test['day'] = df_test.pickup_datetime.dt.dayofweek

df_test['month'] = df_test.pickup_datetime.dt.month
df['distance2'] = np.sqrt((df['pickup_pca0']-df['dropoff_pca0'])**2

                        + (df['pickup_pca1']-df['dropoff_pca1'])**2)

df_test['distance2'] = np.sqrt((df_test['pickup_pca0']-df_test['dropoff_pca0'])**2

                        + (df_test['pickup_pca1']-df_test['dropoff_pca1'])**2)
df['log_trip_duration'] = np.log(df['trip_duration'])

df.head()
df_test.head()
NUM_VARS = ['pickup_pca0', 'pickup_pca1', 'dropoff_pca0', 'dropoff_pca1', 'month', 'hour', 'day', 'distance2']

TARGET = 'log_trip_duration'
num_features = NUM_VARS

X_train = df.loc[:, num_features]

y_train = df[TARGET]

X_test = df_test.loc[:, num_features]

X_train.shape, y_train.shape, X_test.shape
from sklearn.ensemble import RandomForestRegressor

# une analyse par regression semble plus pertinente sur ce modele

training = RandomForestRegressor(n_estimators=19, min_samples_leaf=10, min_samples_split=15, max_features='auto', max_depth=80, bootstrap=True)

training.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(training, X_train, y_train, cv=5, scoring='neg_mean_squared_log_error')
cv_scores
for i in range(len(cv_scores)):

    cv_scores[i] = np.sqrt(abs(cv_scores[i]))

cv_scores
y_test_pred = training.predict(X_test)

y_test_pred[:5]
predection = pd.DataFrame({'id': df_test.id, 'trip_duration': np.exp(y_test_pred)})

# you could use any filename. We choose submission here

predection.to_csv('submission.csv', index=False)
predection.head()