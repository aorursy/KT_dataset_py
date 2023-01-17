# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# IMPORT H2O

import h2o
#Start H2O cluster 

h2o.init()

# Importing H2O AUTOML pacakge 

from h2o.automl import H2OAutoML

# Loading data into H2Oframe . 

train_path = "../input/home-data-for-ml-course/train.csv"

train_ = h2o.upload_file(path = train_path)

#Loading Test dataset 

test_path = '../input/home-data-for-ml-course/test.csv'

test_ = h2o.upload_file(path = test_path)



Id.head()
train_.types
train_.describe()
# Spliting the dataframe in to 3 parts - train , valid , test .

# The argument ratios take 2 values for test split ratio and for valid split ratio and rest as test split ratio with all values less 1.0 and summing 1.0

train ,valid , test = train_.split_frame(ratios =[.7,.15])
valid.describe()
# Removing saleprice and id from train_ 

y = 'SalePrice'

x = train_.columns

x.remove(y)

x.remove('Id')

# Defining our automl model element . 

# max_models will decide maximum no. of models to be trained by automl .

# exclude_algos will specify which algos to be excluded from trainig 

# verbosity is used to see output to help with debug

# n_folds is factor for crossvalidation , by default it is 5 



aml = H2OAutoML(max_models = 10, seed = 10, exclude_algos = ["StackedEnsemble", "DeepLearning"], verbosity="info", nfolds=0)
# Training our model 

aml.train(x = x, y = y, training_frame = train, validation_frame=valid)
# defining lb as leaderboard holder . 

lb = aml.leaderboard

lb.head()
# Predicting values using leader model i.e GBM 

test_pred=aml.leader.predict(test)
test_pred.head()
# Lets check performance of our model . 

aml.leader.model_performance(test)

# Scrapping model ids of from leaderboard 

model_ids = list(aml.leaderboard['model_id'].as_data_frame().iloc[:,0])

model_ids
# We can extract any model from the trained models . 

h2o.get_model(model_ids[0])
out = h2o.get_model(model_ids[0])
out.params
out
out.varimp_plot()
# Let's save a Mojo binary file which is deployable ready 

aml.leader.download_mojo(path = "./")
