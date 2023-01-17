# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train= pd.read_csv('../input/pubg-finish-placement-prediction/train_V2.csv')

test= pd.read_csv('../input/pubg-finish-placement-prediction/test_V2.csv')

submission= pd.read_csv('../input/pubg-finish-placement-prediction/sample_submission_V2.csv')
train.head()
train.tail()
train.describe()
train.info()
train['matchType'].unique()
type(train)
train.shape
test.shape
test.info()
submission.head()
train = pd.get_dummies(train, columns=['matchType'])

matchType_encoding = train.filter(regex='matchType')

matchType_encoding.head()
train.info()
# Turn groupId and match Id into categorical types

train['groupId'] = train['groupId'].astype('category')

train['matchId'] = train['matchId'].astype('category')
# Get category coding for groupId and matchID

train['groupId_cat'] = train['groupId'].cat.codes

train['matchId_cat'] = train['matchId'].cat.codes
# Get rid of old columns

train.drop(columns=['groupId', 'matchId'], inplace=True)



# Lets take a look at our newly created features

train[['groupId_cat', 'matchId_cat']].head()
# Drop Id column, because it probably won't be useful for our Machine Learning algorithm,

# because the test set contains different Id's

train.drop(columns = ['Id'], inplace=True)
# Take sample for debugging and exploration

sample = 500000

df_sample = train.sample(sample)
# Split sample into training data and target variable

df = df_sample.drop(columns = ['winPlacePerc']) #all columns except target

y = df_sample['winPlacePerc'] # Only target variable
# Function for splitting training and validation data

def split_vals(a, n : int): 

    return a[:n].copy(), a[n:].copy()

val_perc = 0.12 # % to use for validation set

n_valid = int(val_perc * sample) 

n_trn = len(df)-n_valid

# Split data

raw_train, raw_valid = split_vals(df_sample, n_trn)

X_train, X_valid = split_vals(df, n_trn)

y_train, y_valid = split_vals(y, n_trn)



# Check dimensions of samples

print('Sample train shape: ', X_train.shape, 

      'Sample target shape: ', y_train.shape, 

      'Sample validation shape: ', X_valid.shape)
# Metric used for the PUBG competition (Mean Absolute Error (MAE))

from sklearn.metrics import mean_absolute_error



# Function to print the MAE (Mean Absolute Error) score

# This is the metric used by Kaggle in this competition

def print_score(m : RandomForestRegressor):

    res = ['mae train: ', mean_absolute_error(m.predict(X_train), y_train), 

           'mae val: ', mean_absolute_error(m.predict(X_valid), y_valid)]

    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)

    print(res)
X_train.info()
from sklearn.ensemble import RandomForestRegressor



# Train basic model

m1 = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features='sqrt',

                          n_jobs=-1)

m1.fit(X_train, y_train)

print_score(m1)
# Turn groupId and match Id into categorical types

test['groupId'] = test['groupId'].astype('category')

test['matchId'] = test['matchId'].astype('category')



# Get category coding for groupId and matchID

test['groupId_cat'] = test['groupId'].cat.codes

test['matchId_cat'] = test['matchId'].cat.codes



test_orig = test.copy()



test.drop(columns=['groupId', 'matchId'], inplace=True)

test.drop(columns = ['Id'], inplace=True)



# Fill NaN with 0 (temporary)

#test_pred.fillna(0, inplace=True)

#test_pred.head()
test_orig.head()
test = pd.get_dummies(test, columns=['matchType'])



matchType_encoding = test.filter(regex='matchType')

matchType_encoding.head()



#test.drop(columns = ['matchType'], inplace=True)
test.info()
# Make submission ready for Kaggle

# We use our final Random Forest model (m3) to get the predictions

predictions = np.clip(a = m1.predict(test), a_min = 0.0, a_max = 1.0)

pred_df = pd.DataFrame({'Id' : test_orig['Id'], 'winPlacePerc' : predictions})

# Last check of submission

print('Head of submission: ')

display(pred_df.head())

print('Tail of submission: ')

display(pred_df.tail())
# Create submission file

pred_df.to_csv("submission.csv", index=False)