# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

train.head(1)
test.head(1)
y=train['trip_duration']
X=train.drop('trip_duration',axis=1)
X.head(1)
X.shape
y.shape
test.head(1)
test.shape
mapping = {'N':0, 'Y':1}

X = X.replace({'store_and_fwd_flag':mapping})

test=test.replace({'store_and_fwd_flag':mapping})
X=X.drop('pickup_datetime',1)

test=test.drop('pickup_datetime',1)

X=X.drop('dropoff_datetime',1)

X=X.drop('id',1)

sub=test['id']

test=test.drop('id',1)
test.head(1)
X.head(1)
y.head(1)
train['vendor_id'].value_counts()
from sklearn.ensemble import *
Reg=RandomForestRegressor()
Reg.fit(X,y)
predictions=Reg.predict(test)
submission=pd.DataFrame({'id':sub,'trip_duration':predictions})
submission.head()
submission.to_csv('subRandomForsts.csv',index=False)
Ada=AdaBoostRegressor()
Ada.fit(X,y)
predictions1=Ada.predict(test)
submission1=pd.DataFrame({'id':sub,'trip_duration':predictions})
submission.head()
submission.to_csv('Adaboost.csv',index=False)