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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
# Reading our test and train csv 

data_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

data_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

# Understanding the our data size by observing the shape  

print ("data_train shape", data_train.shape)

print ("data_test shape", data_test.shape)
# NAN values check

print ('Nan values present in training set',data_train.isna().sum().sum(),'\n',

       'Nan values present in testing set',data_test.isna().sum().sum())

#To concatinate we need to drop Saleprice column from training data set 

X = data_train.drop(columns=['SalePrice'],axis=1)

y = data_train['SalePrice']
data = pd.concat(objs=[X,data_test],axis=0)

data.shape
one_hot_encoding = pd.get_dummies(data)

one_hot_encoding.shape
 # Taking care of missing data

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)

imputer = imputer.fit(one_hot_encoding)

imputed_data= imputer.transform(one_hot_encoding)

imputed_data = pd.DataFrame(imputed_data)
imputed_data.isna().sum().sum()
#spliting our data back into training and test sets 

training = imputed_data[:len(data_train)]

test = imputed_data[len(data_train):]
print (training.shape ,test.shape)
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(n_estimators=6000,

                                learning_rate=0.01,

                                max_depth=4,

                                max_features='sqrt',

                                min_samples_leaf=15,

                                min_samples_split=10,

                                loss='huber',

                                random_state=42)
gbr.fit(training,y)

gbr.score
pred = gbr.predict(test)
pred
my_submission = pd.DataFrame({'Id': test[0].astype('int64'), 'SalePrice':pred})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)
my_submission