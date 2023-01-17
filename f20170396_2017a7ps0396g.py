# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from google.colab import drive

drive.mount('/content/drive')



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



df = pd.read_csv('/content/drive/My Drive/train.csv')

y = df['label']

df = df.drop('label',axis=1)

df = df.drop('id',axis=1)



to_drop=[]

for i in df.columns:

  if(df[i].nunique()==1):

    to_drop.append(i)

    print(i,":",df[i].nunique())



df2 = df.drop(to_drop,axis=1)

df2.shape



from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(df2, y, test_size=0.25,random_state=1)



print(x_train.shape)

print(x_val.shape)



from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor

regr = AdaBoostRegressor(random_state=0, n_estimators=300,learning_rate=1.0, loss='square', base_estimator = DecisionTreeRegressor(max_depth=None))

regr.fit(x_train, y_train)  



from sklearn import metrics

y_pred2 = regr.predict(x_val)

print("RMSE:",metrics.mean_squared_error(y_val, y_pred2, squared=False))



test_df = pd.read_csv('/content/drive/My Drive/test.csv')

test_id = test_df['id']

test_df = test_df.drop('id',axis=1)

test_df = test_df.drop(to_drop,axis=1)

test_df.shape

x_test=test_df



ans=regr.predict(x_test)



ansdf=pd.DataFrame(ans)



final = pd.concat([test_id,ansdf],axis=1)

final.columns = ["id","label"]



final.to_csv('final8.csv',index=False)