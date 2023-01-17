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
import pandas as pd
import numpy as np

train = pd.read_csv('/kaggle/input/mlregression-cabbage-price/train_cabbage_price.csv')
test = pd.read_csv('/kaggle/input/mlregression-cabbage-price/test_cabbage_price.csv')

x = train.values[:,:-1]
y = train.values[:,[-1]]

test = test.values[:,:]
from sklearn.neighbors import KNeighborsRegressor

reg = KNeighborsRegressor(n_neighbors=3, weights = "distance")

reg.fit(x,y)
pred = reg.predict(test)
pred
submit=pd.read_csv('/kaggle/input/mlregression-cabbage-price/sample_submit.csv')
for i in range(len(pred)):
  submit['Expected'][i] = pred[i]
submit
submit.to_csv('sample_submit.csv',index=False)