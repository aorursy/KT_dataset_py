# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

PATH = "../input/wec_24/" # unfortunately the creators of this competition have made a sub folder inside input

import os
print(os.listdir(PATH))

import pandas as pd
# Any results you write to the current directory are saved as output.
import h2o
h2o.init()
from h2o.automl import H2OAutoML
train = pd.read_csv(PATH + 'train.csv')
test = pd.read_csv(PATH + 'test.csv')
ss = pd.read_csv(PATH + 'SampleSubmission.csv')
hT = h2o.H2OFrame(train.drop('Id', axis = 1)) # we don't want to train or predict using 'Id'
hTest = h2o.H2OFrame(test.drop('Id', axis = 1))

cat_columns = ['Label', 'land', 'logged_in', 'is_host_login', 'is_guest_login']
for col in cat_columns:
    hT[col] = hT[col].asfactor() # categorical 
    if col != 'Label': hTest[col] = hTest[col].asfactor() # categorical, but label is not available in test

x = hT.columns
y ='Label'
x.remove(y)

aml = H2OAutoML(nfolds=0, max_runtime_secs = 60*60, sort_metric = 'logloss', seed=79)
aml.train(x = x, y = y, training_frame=hT)
aml.leaderboard
preds = aml.leader.predict(hTest) #
preds = preds.as_data_frame()
ss.iloc[:, 1:] = preds.iloc[:, 1:]
ss.to_csv('sub.csv', index = False)