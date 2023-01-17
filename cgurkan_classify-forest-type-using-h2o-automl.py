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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import h2o

from h2o.estimators.gbm import H2OGradientBoostingEstimator

from h2o.automl import H2OAutoML

h2o.init()
train_df = h2o.import_file('../input/learn-together/train.csv')

test_df = h2o.import_file('../input/learn-together/test.csv')
test_df.shape
train_df["Cover_Type"].describe()
#Drop Id columns

train_df = train_df.drop('Id', axis = 1)

#test_ids = test_df["Id"].squeeze() 

#test_df = test_df.drop('Id', axis = 1)



# Make target as categorical

train_df['Cover_Type'] = train_df['Cover_Type'].asfactor()



#Predictor Columns

x_col = train_df.columns

x_col = x_col.remove('Cover_Type')



y_col = 'Cover_Type'



#Split data into training and validation

d = train_df.split_frame(ratios = [0.8], seed = 42)

hf_train = d[0] # using 80% for training

hf_valid = d[1] # rest 20% for testing
aml = H2OAutoML(seed = 42, max_models=10, max_runtime_secs=1800, verbosity='info')

aml.train(x = x_col, y = y_col, training_frame = hf_train, validation_frame=hf_valid)
print(aml.leaderboard)
print(aml.leader)
from sklearn.metrics import accuracy_score, f1_score



preds = aml.leader.predict(hf_valid)

accuracy_score(preds['predict'].as_data_frame(), hf_valid['Cover_Type'].as_data_frame())
h2o.save_model(aml.leader)
#Output

preds = aml.leader.predict(test_df)



# Save test predictions to file

output = pd.DataFrame({'Id': test_df["Id"].as_data_frame().squeeze(),

                       'Cover_Type': preds['predict'].as_data_frame().squeeze()})



output.to_csv('submission.csv', index=False)