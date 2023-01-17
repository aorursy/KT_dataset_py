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

import os

import datetime

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
train = pd.read_csv("../input/into-the-future/train.csv")

test = pd.read_csv("../input/into-the-future/test.csv")
train.head()
train.info()
train['time'] =  pd.to_datetime(train['time'])

test['time'] =  pd.to_datetime(test['time'])
train = train.set_index('time')

test = test.set_index('time')
ids = test['id'].values
del train['id']

del test['id']
train.columns = ['observation', 'prediction']

test.columns = ['observation']
from statsmodels.tsa.vector_ar.vecm import coint_johansen
coint_johansen(train,-1,1).eig
#fit the model

from statsmodels.tsa.vector_ar.var_model import VAR
model = VAR(endog=train)

model_fit = model.fit()
# make prediction on validation

prediction = model_fit.forecast(model_fit.y, steps=len(test))
prediction
#converting predictions to dataframe

pred = pd.DataFrame(index=range(0,len(prediction)),columns=['observation', 'feature_2'])

for j in range(0,2):

    for i in range(0, len(prediction)):

       pred.iloc[i][j] = prediction[i][j]
pred['id'] = ids
pred
#check rmse

cols = ['observation']

for i in cols:

    print('rmse value for', i, 'is : ', np.sqrt(mean_squared_error(pred[i], test[i])))
#The rmse value is really small, hence the predictions that we have obtained from the model can be used
#final predictions

del pred['observation']
pred
pred.to_csv("result.csv", index=False)