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
train=pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv')
test=pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/test_V2.csv')
train.head()
train['winPlacePerc'].fillna(0.47,inplace=True)
test.info()
train.shape
test.shape
#train['matchType']
matchType=pd.get_dummies(train['matchType'])
matchtype=pd.get_dummies(test['matchType'])
train=pd.concat([train,matchType],axis=1)
test=pd.concat([test,matchtype],axis=1)
test.isna().sum()
from sklearn.model_selection import train_test_split
X=train.drop(['Id','groupId','matchId','matchType','winPlacePerc'],axis=1)
y=train['winPlacePerc']
test_model=test.drop(['Id','groupId','matchId','matchType'],axis=1)
#test.head()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
test_model.columns
X_train.columns
from sklearn.linear_model import LinearRegression
model_lr=LinearRegression()
model_lr.fit(X_train,y_train)
predictions=model_lr.predict(test_model)
sub=pd.DataFrame(test['Id'],columns=['Id'])
sub['WinPlacePerc']=np.array(predictions)
sub.head()
sub.to_csv('lr.csv',index=False)
