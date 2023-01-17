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
df1=pd.read_csv('../input/train_V2.csv')

df1.info()
X=df1.iloc[:,:-1]

X.head()
y=df1.iloc[:,25:26]

y.head()
from xgboost import XGBRegressor

xgb=XGBRegressor()

xgb.fit(X,y)
df2=pd.read_csv('../input/test_V2.csv')

X_test=df2.iloc[:,:]
y_pred=xgb.predict(X_test)
y_pred=pd.DataFrame(y_pred)

y_pred.head()
df3=df2.Id.astype(str)

df3=pd.DataFrame(df3)

df3.head()
df3=pd.concat([df3,y_pred],axis=1)

df3.head()
col=['Id','winPlacePerc']

df3.columns=col

df3.head()
df3.to_csv('Solution1.csv',index=False)