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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline


data_path = '../input/hr_data.csv' # Path to data file

data = pd.read_csv(data_path) 

data.head()



from sklearn.neural_network import MLPRegressor

X_columns=['last_evaluation','number_project','average_montly_hours','time_spend_company']

y_column='satisfaction_level'
from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error
K=3

kf = KFold(n_splits = K)
ann_reg = MLPRegressor()
kfold_splits =kf.split(data)

scores=[]
for train_indexes, test_indexes in kfold_splits:

    print("\\Increment %d" %counter)

    print(train_indexes)

    print("Training indexes size %d" %train_indexes.size)

    print(test_indexes)

    print("Testing indexes size %d" %test_indexes.size)

    

    train = data.iloc[train_indexes]

    print (train.head(5))

    test = data.iloc[test_indexes]

    print(test.head(5))

    

    #split into dependednt and independent variables

    train_X=train[X_columns]

    train_y=train[y_column]

    

    test_X = test[X_columns]

    test_y= test[y_column]

    

    #train the model

    ann_reg.fit(train_X,train_y)

    

    #retrieve the predicted values

    test_predicted = ann_reg.predict(test_X)

    

    #calculate RMSE

    

    score = np.sqrt(mean_squared_error(test_y,test_predicted))

    

    scores.append(score)

    
scores

np.std(scores)
print ("Average RSME for model = %f" %np.mean(scores))
from sklearn.model_selection import cross_val_score

ann_reg = MLPRegressor()

X_data = data[X_columns]

Y_data = data[y_column]

#retrieve scores

scores= cross_val_score(ann_reg,X_data,Y_data,cv=K)
print("Accuracy(mean score): %0.5f"% scores.mean())

print("and the 95%% confidence interval:(+/- %0.5f)" %(scores.std()*2))