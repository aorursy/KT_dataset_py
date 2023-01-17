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
train = pd.read_csv('/kaggle/input/hr-analysis/train.csv')
test =  pd.read_csv('/kaggle/input/hr-analysis/test.csv')
sample =  pd.read_csv('/kaggle/input/hr-analysis/sample_submission.csv')

import h2o
h2o.init()
train1 = h2o.H2OFrame(train)
test1 = h2o.H2OFrame(test)
train1.columns
y = 'target'
x = train1.col_names
x.remove(y)
train1['target'] = train1['target'].asfactor()
train1['target'].levels()
from h2o.automl import H2OAutoML
aml = H2OAutoML(max_models = 20,max_runtime_secs=2000, seed = 42)
aml.train(x = x, y = y, training_frame = train1)
preds = aml.predict(test1)
ans=h2o.as_list(preds) 

sample['target'] = ans['predict']
sample.to_csv('Solution1.csv',index=False)
