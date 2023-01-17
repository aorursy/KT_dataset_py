# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

train_df['first_active_month'] = pd.to_datetime(train_df['first_active_month'])
train_df['first_active_month'] = pd.to_numeric(train_df['first_active_month'])

test_df['first_active_month'] = pd.to_datetime(test_df['first_active_month'])
test_df['first_active_month'] = pd.to_numeric(test_df['first_active_month'])
from sklearn.ensemble import RandomForestRegressor
X = train_df.drop(['card_id','target'],axis=1)
y = train_df['target']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.metrics import mean_squared_error
import numpy as np

rf = RandomForestRegressor(max_depth=2,n_estimators=200)
rf.fit(X_train,y_train)

pred = rf.predict(X_test)
print(np.sqrt(mean_squared_error(pred,y_test)))
rf.fit(X,y)
index = test_df['card_id']
test = test_df.drop(['card_id'],axis=1)

pred = rf.predict(test)
dfdict = {}
dfdict["card_id"]=index
dfdict["target"]=pred

df=pd.DataFrame(dfdict)
df.to_csv("solution.csv",index=False,columns=["card_id","target"])
