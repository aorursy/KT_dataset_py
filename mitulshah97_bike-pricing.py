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
import numpy as np
import pandas as pd

test1 = pd.read_csv("../input/test.csv")
train = pd.read_csv("../input/train.csv")
test=test1.drop(['datetime'],axis=1)
from sklearn.ensemble import BaggingRegressor
X = train.drop(['count','datetime','casual','registered'],axis=1)
y = train['count']
model_rf = BaggingRegressor(random_state=1211)
model_rf.fit( X , y )
y_pred = model_rf.predict(test)
y_pred1=np.round(y_pred,decimals=0)
a= test1['datetime']
count = pd.Series(y_pred1)

submit = pd.concat([a,count],axis=1, ignore_index=True)
submit.columns=['datetime','count']
submit.to_csv("submitbike.csv",index=False)