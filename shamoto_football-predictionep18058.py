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
train_df = pd.read_csv('/kaggle/input/1056lab-football-players-value-prediction/train.csv', index_col=0)

test_df = pd.read_csv('/kaggle/input/1056lab-football-players-value-prediction/test.csv', index_col=0)
train_df
test_df
train_df.dtypes
all_df = pd.concat([train_df.drop('value_eur', axis=1), test_df])

all_df
all_df = all_df.replace('?', np.NaN)

all_df
columns = all_df.columns

for c in columns:

    all_df[c] = pd.to_numeric(all_df[c], errors='ignore')

all_df
all_df.dtypes
columns = all_df.columns

for c in columns:

    if all_df[c].isna().any():

        if all_df[c].dtypes != np.object:

            median = all_df[c].median()

            all_df[c] = all_df[c].replace(np.NaN, median)

        else:

            mfv = all_df[c].mode()[0]

            all_df[c] = all_df[c].replace(np.NaN, mfv)
columns = all_df.columns

for c in columns:

    if all_df[c].dtypes == np.object:

        dummy_df = pd.get_dummies(all_df[[c]])

        all_df = pd.concat([all_df, dummy_df], axis=1)

        all_df = all_df.drop(c, axis=1)

all_df
X_train = all_df[:len(train_df)].to_numpy()

y_train = train_df['value_eur'].to_numpy()



X_test = all_df[len(train_df):].to_numpy()
from sklearn.ensemble import RandomForestRegressor



model = RandomForestRegressor()

model.fit(X_train, y_train)
p_test = model.predict(X_test)
submit_df = pd.read_csv('/kaggle/input/1056lab-football-players-value-prediction/sampleSubmission.csv', index_col=0)

submit_df['value_eur'] = p_test

submit_df
submit_df.to_csv('submission.csv')