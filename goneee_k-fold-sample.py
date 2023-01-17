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
data_path = '../input/hr_data.csv'

data  = pd.read_csv(data_path)
from sklearn.neural_network import  MLPRegressor
X_columns = ['last_evaluation','number_project','average_montly_hours','time_spend_company']

y_column = 'satisfaction_level'
from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error
K = 3 # Using K=3 folds/groups

kf = KFold(n_splits=K) # create in instance of the K-Fold object
kf
ann_reg = MLPRegressor() # created the model
kfold_splits = kf.split(data2) # split our data into K pieces randomly

scores = [] # list to store all the evaluation scores for each iteration



# applying K-Fold

counter = 1

for train_indexes, test_indexes in kfold_splits:

    print("\n\n\nIteration %d" % counter)

    print("Training indexes  %d " % train_indexes.size )

    print(train_indexes)

    print("Testing indexes size %d " % test_indexes.size)

    print(test_indexes)

    counter = counter + 1

    # Now that we have indexes let us extract the rows/observations/cases with the data for 

    # train and test

    

    train = data.iloc[train_indexes]

    print(train.head(5))

    test = data.iloc[test_indexes]

    print(test.head(5))

    #break

    

    # split into dependent and independent variables

    train_X = train[X_columns]

    train_y = train[y_column]

    

    test_X = test[X_columns]

    test_y = test[y_column]

    

    # train the model

    ann_reg.fit(train_X,train_y)

    

    # retrieve the predicted values

    test_predicted = ann_reg.predict(test_X)

    # calculate RMSE

    score = np.sqrt(  mean_squared_error(test_y,test_predicted)   )

    # add to our list of scores to be averaged later

    scores.append(score)

    #break