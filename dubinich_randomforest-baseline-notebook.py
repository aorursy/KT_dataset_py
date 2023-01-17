# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_val_score

from  sklearn.ensemble import RandomForestRegressor

plt.style.use('fivethirtyeight')

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/mldub-comp1/train_data.csv')

test = pd.read_csv('/kaggle/input/mldub-comp1/test_data.csv')

sample_sub = pd.read_csv('/kaggle/input/mldub-comp1/sample_sub.csv')
train.head()
test.head()
train.info()
numerics = ['photos', 'videos', 'comments']



for feature in numerics:

    plt.figure()

    train[feature].hist()

    plt.title(feature)

    plt.show()
# Checking minimum of each feature to safely use log transformation.

for feature in numerics:

    print('{} min value: {}'.format(feature, train[feature].min()))
# To use log the feature must be a positive number, so let's take abs and log then.

# Do not forget to apply the same transformation to test data as well!

# We are safe to use abs, because negative number of videos probably indicates parsing

# error on data preparation stage.



for feature in numerics:

    train[feature] = np.log1p(np.abs(train[feature]))

    test[feature] = np.log1p(np.abs(test[feature]))
for feature in numerics:

    plt.figure()

    train[feature].hist()

    plt.title(feature)

    plt.show()
plt.figure()

train['target_variable'].hist()

plt.title('Target distribution')

plt.show()



plt.figure()

train['target_variable'].apply(lambda x: np.log1p(x-train['target_variable'].min())).hist()

plt.title('Log-transformed target distribution')

plt.show()
train['about'].head()
train['about_len'] = train['about'].fillna('').apply(len)

test['about_len'] = test['about'].fillna('').apply(len)
train['about_len'].hist(alpha=0.4)

test['about_len'].hist(alpha=0.4)
plt.figure()

plt.scatter(train['about_len'], train['target_variable'])

plt.title('Target vs. about section length')

plt.show()
train['status'].unique()
test['status'].unique()
for df in [train, test]:

    df.loc[~df['status'].isin(['Deadpool', 'Submission', 'Confirmed']), 'status'] = 'Unknown'
# Check results

train['status'].unique()
# Here we use manual mapping, you can use other tools instead.

mapping = {'Deadpool': 0,

           'Submission': 1,

           'Confirmed': 2,

           'Unknown': -1}



train['status'] = train['status'].map(mapping)

test['status'] = test['status'].map(mapping)
selected_features = ['videos', 'comments', 'photos', 'about_len', 'status']



X_train = train[selected_features]

y_train = train['target_variable']



X_test = test[selected_features]
# Creating Kfold splitter object.

cv = KFold(3, shuffle=True, random_state=42)
model = RandomForestRegressor(n_estimators=10,

                              random_state=42,

                              n_jobs=-1,

                              verbose=2)
cv_results = cross_val_score(model,

                             X_train,

                             y_train,

                             cv=cv,

                             scoring='neg_mean_squared_error')
# There is no RMSE score function in sklearn, so we use the MSE option

# and take the square root then. You can check that it is the same :)

np.sqrt([-x for x in cv_results])
model.fit(X_train, y_train)
preds = model.predict(X_test)
submission = sample_sub

submission['target_variable'] = preds
submission.to_csv('submission.csv', index=False)