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

from sklearn.neighbors import KNeighborsRegressor
train_data = pd.read_csv('/kaggle/input/mlregression-cabbage-price/train_cabbage_price.csv', header=None, skiprows=1, usecols=range(1,6))

test_data = pd.read_csv('/kaggle/input/mlregression-cabbage-price/test_cabbage_price.csv')

submit = pd.read_csv('/kaggle/input/mlregression-cabbage-price/sample_submit.csv')
train_data
X_train = train_data.loc[:,1:4]

y_train = train_data.loc[:,5]



knn = KNeighborsRegressor(n_neighbors=100, weights="distance")
print(X_train.shape)

print(y_train.shape)
knn.fit(X_train, y_train)
X_test = test_data.loc[:,test_data.keys()[1:,]]
X_test
predict = knn.predict(X_test)
predict = predict.astype(np.int32)

id = np.array([i for i in range(predict.shape[0])]).reshape(-1,1).astype(np.int32)

res = np.hstack([id, predict.reshape(-1,1)])
submission = pd.DataFrame(res,columns=["ID","Expected"])

submission.to_csv("sample_submit.csv",mode='w',header=True,index=False)
submission