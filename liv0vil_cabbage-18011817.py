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



train = pd.read_csv("../input/mlregression-cabbage-price/train_cabbage_price.csv")

test = pd.read_csv("../input/mlregression-cabbage-price/test_cabbage_price.csv")
train['year'] = train['year'] % 10000 / 1000

test['year'] = test['year'] % 10000 / 1000



X_train = train.iloc[:,:-1]

y = train.iloc[:,-1]
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)

X_test_std = sc.transform(test)
from sklearn.neighbors import KNeighborsRegressor

regressor = KNeighborsRegressor(n_neighbors =60, p = 2, weights='distance') 

regressor.fit(X_train_std, y) 
y_train_pred=regressor.predict(X_train_std) 

y_test_pred=regressor.predict(X_test_std) 



print('Error(training sample): %d' %(abs(y-y_train_pred)).sum()) 
submit = pd.read_csv("../input/mlregression-cabbage-price/sample_submit.csv")
for i in range(len(y_test_pred)):

  submit['Expected'][i] = y_test_pred[i]

submit=submit.astype(np.int32)

submit.to_csv('submit.csv', mode='w', header= True, index= False)