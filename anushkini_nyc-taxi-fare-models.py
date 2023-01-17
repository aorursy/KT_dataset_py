# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#package imports

import pandas as pd

from sklearn import preprocessing

from sklearn.model_selection import KFold

from sklearn import linear_model

from sklearn.externals import joblib

import numpy as np
#read the test and train sets

train_df= pd.read_feather('../input/kernel318ff03a29/nyc_taxi_data_raw.feather')

test_df = pd.read_feather('../input/kernel318ff03a29/test_feature.feather')
#examine the dataset's fist 5 rows

train_df.head()
#examine the dataset

test_df.head()
#for training the model on a part of the data... we have used the whole dataset in the cells below..

chunk = 20_000_000

train_df = train_df[:chunk]
#select all rows, and all columns after the second column

X = train_df.iloc[:,3:]

#target variable

y = train_df['fare_amount']

#select all rows, and all columns after the second column

X_test = test_df.iloc[:,2:]

#reorder the columns

X_test = X_test[X.columns]

X.head()

X_test.head()

import gc

gc.collect()
# fit a normal liner regression model to this dataset.

#Note: this library uses the closed form expression for the parameters and not gradient descent

model = linear_model.LinearRegression()

model.fit(X,y)
print("R2 value of model",model.score(X,y))
X.head()
X_test.head()
y_test = model.predict(X_test)

joblib.dump(model, 'LinearRegression')

 
#create a dataframe in the submission format

holdout = pd.DataFrame({'key': test_df.key, 'fare_amount': y_test})

#write the submission file to output

holdout.to_csv('submission.csv', index=False)
holdout.head()
len(holdout)
model = linear_model.Lasso(normalize = True)

gc.collect()

model.fit(X,y)
print("R2 value of model",model.score(X,y))

y_test = model.predict(X_test)

joblib.dump(model, 'LassoRegression')

holdout = pd.DataFrame({'key': test_df.key, 'fare_amount': y_test})

#write the submission file to output

holdout.to_csv('submission_lasso.csv', index=False)
model = linear_model.Ridge(normalize = True)

gc.collect()

model.fit(X,y)

print("R2 value of model",model.score(X,y))

y_test = model.predict(X_test)

joblib.dump(model, 'RidgeRegression')

holdout = pd.DataFrame({'key': test_df.key, 'fare_amount': y_test})

#write the submission file to output

holdout.to_csv('submission_ridge.csv', index=False)
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(max_depth=4, random_state=0, n_estimators=100)

gc.collect()

model.fit(X,y)

y_test = model.predict(X_test)

joblib.dump(model, 'RandomForest')

holdout = pd.DataFrame({'key': test_df.key, 'fare_amount': y_test})

#write the submission file to output

holdout.to_csv('submission_rf.csv', index=False)