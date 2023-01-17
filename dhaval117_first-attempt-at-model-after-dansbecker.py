# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

file_path = "../input/train.csv"

training_data_original = pd.read_csv(file_path)
#removing SalePrice and Id from training data and setting sale price as entity to be predicted

prediction_target = training_data_original.SalePrice

training_data_numerical = training_data_original.select_dtypes(exclude=['object'])

filtered_training_data = training_data_numerical.drop(['SalePrice','Id'],axis=1)
#filling empty data with imputer to generate final training data

from sklearn.preprocessing import Imputer

my_imputer = Imputer()

final_training_data = my_imputer.fit_transform(filtered_training_data)
#using RandomForestRegressor as model with max_leaf_node set as 79

from sklearn.ensemble import RandomForestRegressor

data_model = RandomForestRegressor(max_leaf_nodes = 79,random_state = 0)

data_model.fit(final_training_data,prediction_target)
#filtering test data similar to training data

file_path2 = "../input/test.csv"

test_data_original = pd.read_csv(file_path2)

test_data_numerical = test_data_original.select_dtypes(exclude=['object'])

filtered_test_data = test_data_numerical.drop(['Id'],axis=1)

final_test_data = my_imputer.fit_transform(filtered_test_data) 

final_predictions = data_model.predict(final_test_data)
#generating CSV file for submission

my_submission = pd.DataFrame({'Id': test_data_original.Id, 'SalePrice': final_predictions})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)