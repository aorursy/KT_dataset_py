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
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd
train = pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/train.csv')
test = pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/test.csv')
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()



# Train a model

rf.fit(X=train[['store', 'item']], y=train['sales'])
# Get predictions for the test set

test['sales'] = rf.predict(test[['store', 'item']])
# Write test predictions using the sample_submission format

test[['id', 'sales']].to_csv('/kaggle/working/kaggle_submission.csv', index=False)