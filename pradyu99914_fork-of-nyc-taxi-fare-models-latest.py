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

test_df1 = train_df[53000000:]

y_test_actual = train_df.fare_amount.iloc[53000000:]

train_df = train_df[:chunk]

#select all rows, and all columns after the second column

X = train_df.iloc[:,3:]

#target variable

y = train_df['fare_amount']

#select all rows, and all columns after the second column

X_test = test_df.iloc[:,2:]

#reorder the columns

X_test = X_test[X.columns]

test_df1 = test_df1[X.columns]

X.head()

X_test.head()

import gc

gc.collect()

test_df.head()
gc.collect()
from sklearn.metrics import mean_squared_error

from math import sqrt

from tqdm import tqdm

import matplotlib.pyplot as plt

minrms = float('inf')

minrmsalpha = -1

rmserrs = []

for i in range(0, 6):

    gc.collect()

    model = linear_model.Lasso(normalize = True, alpha = 10**(-i))

    gc.collect()

    model.fit(X,y)

    print("R2 value of model",model.score(X,y))

    y_test = model.predict(X_test)

    joblib.dump(model, 'LassoRegression')

    holdout = pd.DataFrame({'key': test_df.key, 'fare_amount': y_test})

    #write the submission file to output

    holdout.to_csv('submission_lasso.csv', index=False)



    y_test1 = model.predict(test_df1)

    #create a dataframe in the submission format

    holdout = pd.DataFrame({'fare_amount': y_test1})

    #write the submission file to output

    holdout.to_csv('testlasso.csv', index=False)

    rms = sqrt(mean_squared_error(y_test_actual, y_test1))

    print("rms for ", i, rms)

    rmserrs.append(rms)

    del model

    if rms<minrms:

        minrms = rms

        minrmsalpha = i

plt.plot(range(0,6),rmserrs)

plt.title("Grid search for lasso regression")

plt.xlabel("alpha (10^-x)")

plt.ylabel("RMSE")
from sklearn.metrics import mean_squared_error

from math import sqrt

minrms = float('inf')

minrmsalpha = -1

rmserrs = []

for i in range(0, 6):

    gc.collect()

    model = linear_model.Ridge(normalize = True, alpha = 10**(-i))

    gc.collect()

    model.fit(X,y)

    print("R2 value of model",model.score(X,y))

    y_test = model.predict(X_test)

    joblib.dump(model, 'Ridgeregression')

    holdout = pd.DataFrame({'key': test_df.key, 'fare_amount': y_test})

    #write the submission file to output

    holdout.to_csv('submission_ridge.csv', index=False)



    y_test1 = model.predict(test_df1)

    #create a dataframe in the submission format

    holdout = pd.DataFrame({'fare_amount': y_test1})

    #write the submission file to output

    holdout.to_csv('testridge.csv', index=False)

    rms = sqrt(mean_squared_error(y_test_actual, y_test1))

    print("rms for ", i, rms)

    rmserrs.append(rms)

    del model

    if rms<minrms:

        minrms = rms

        minrmsalpha = i

plt.plot(range(0, 6),rmserrs)

plt.title("Grid search for ridge regression")

plt.xlabel("alpha (10^-x)")

plt.ylabel("RMSE")