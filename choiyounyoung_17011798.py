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
train=pd.read_csv('../input/mlregression-cabbage-price/train_cabbage_price.csv')

test=pd.read_csv('../input/mlregression-cabbage-price/test_cabbage_price.csv')

submit=pd.read_csv('../input/mlregression-cabbage-price/sample_submit.csv')
X=train.drop('avgPrice',axis=1).drop('year',axis=1)

x=test.drop('year',axis=1)

y=train['avgPrice']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.18, random_state=1)
from sklearn.neighbors import KNeighborsRegressor

regressor = KNeighborsRegressor(n_neighbors=15, weights="distance")

regressor.fit(X_train,y_train)

y_train_pred=regressor.predict(X_train)

y_test_pred=regressor.predict(X_test)

print('Miss training samples: %d' %(y_train!=y_train_pred).sum())

print('Miss test samples: %d'%(y_test!=y_test_pred).sum())
pred=regressor.predict(x)

for i in range(len(pred)):

    submit['Expected'][i]=pred[i]

submit=submit.astype(np.int32)

submit.to_csv('submit.csv', mode='w', header=True, index=False)