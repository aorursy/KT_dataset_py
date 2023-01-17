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

import numpy as np

import gc 

from tqdm import tqdm

from sklearn.neighbors import KNeighborsRegressor
#read the test and train sets

train_df= pd.read_feather('../input/kernel318ff03a29/nyc_taxi_data_raw.feather')

test_df = pd.read_feather('../input/kernel318ff03a29/test_feature.feather')
#examine the dataset's fist 5 rows

print(len(train_df))

train_df.head()
'''#examine the dataset

test_df.head()

#select all rows, and all columns after the second column

X = train_df.iloc[:,3:]

#target variable

y = train_df['fare_amount']'''
#select all rows, and all columns after the second column

X_test = test_df.iloc[:,2:]

#reorder the columns

Xt = train_df.iloc[:5,3:]

X_test = X_test[Xt.columns]

Xt.head()

X_test.head()

import gc

gc.collect()
X_test.head()
len(train_df)
knnregressoroutputs = []

knnregressoroutputstest = []

X_test2 = train_df.iloc[53000000:,:]

X_test2 = X_test2[Xt.columns]

for i in tqdm(range(len(train_df)//1000000)):

    neigh = KNeighborsRegressor(n_neighbors=2)

    df_chunk = train_df.iloc[i*10**6:(i+1)*10**6, :]

    X = df_chunk.iloc[:,3:]

    #target variable

    y = df_chunk['fare_amount']

    neigh.fit(X,y)

    y_test = neigh.predict(X_test)

    y_test1 = neigh.predict(X_test2)

    knnregressoroutputs.append(y_test)

    knnregressoroutputstest.append(y_test1)

    neigh = 0

    gc.collect()



res = knnregressoroutputs[0]

for i in knnregressoroutputs[1:]:

    res+=i

res/=len(knnregressoroutputs)

holdout = pd.DataFrame({'key': test_df.key, 'fare_amount': res})

#write the submission file to output

holdout.to_csv('submission.csv', index=False)



res = knnregressoroutputstest[0]

for i in knnregressoroutputstest[1:]:

    res+=i

res/=len(knnregressoroutputstest)

holdout = pd.DataFrame({'fare_amount': res})

#write the submission file to output

holdout.to_csv('test_result.csv', index=False)