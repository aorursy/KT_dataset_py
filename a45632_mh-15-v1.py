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
train = pd.read_csv('../input/melanoma-tumor-size-prediction-machinehack/Train.csv')
test = pd.read_csv('../input/melanoma-tumor-size-prediction-machinehack/Test.csv')
sub = pd.read_csv('../input/melanoma-tumor-size-prediction-machinehack/sample_submission.csv')
train.head(5)
import h2o
print(h2o.__version__)
from h2o.automl import H2OAutoML

h2o.init(max_mem_size='16G')
htrain = h2o.H2OFrame(train)
htest = h2o.H2OFrame(test)
x = ['mass_npea', 'size_npear', 'malign_ratio', 'damage_size',
       'exposed_area', 'std_dev_malign', 'err_malign', 'malign_penalty',
       'damage_ratio']
y = 'tumor_size'
#train[y] = train[y]#.asfactor()
aml = H2OAutoML(max_models=50, seed=666, sort_metric = "rmse", max_runtime_secs=1800)# ,exclude_algos = ["DeepLearning"]
aml.train(x=x, y=y, training_frame=htrain)#, fold_column='fold_column')
# View the AutoML Leaderboard
lb = aml.leaderboard
lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)
# The leader model is stored here
aml.leader
preds = aml.leader.predict(htest)
preds = preds.as_data_frame()
sub['tumor_size'] = preds.astype(float)
sub.head(5)
sub.describe()
#sub.to_csv('MH15_baseline_v1ka.csv')
sub.to_csv('MH15_baseline_v1kb.csv')
