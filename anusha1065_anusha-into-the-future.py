# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('../input/into-the-future/train.csv')

df.head()
df.shape
df.dtypes
df['time']=pd.to_datetime(df['time'])
df.dtypes
df['time']=df['time'].astype('category').cat.codes

print(df.dtypes)

df.head()

from sklearn.model_selection import train_test_split

Y=df['feature_2']

X=df.drop('feature_2',axis=1)

# print(X,Y)

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

print(x_train.shape,x_test.shape)
import numpy as np

from sklearn.linear_model import SGDRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn import preprocessing

from sklearn.metrics import accuracy_score

# x_train,y_train=np.asarray(x_train),np.asarray(y_train)

# model=SGDRegressor(shuffle=False, max_iter=5000,learning_rate='optimal',random_state=0)

model = RandomForestRegressor(n_estimators=1000, max_depth=500, n_jobs=-1, random_state=0)

model.fit(x_train,y_train)

y_predict=model.predict(x_test)

print(y_predict)
df1=pd.read_csv('../input/into-the-future/test.csv')

df1.head()
df1['time']=pd.to_datetime(df1['time'])
df1['time']=df1['time'].astype('category').cat.codes

print(df1.dtypes)

df1.head()

test_predict=model.predict(df1)

print(test_predict)
os.chdir(r'/kaggle/working')
import numpy as np

import pandas as pd

prediction = pd.DataFrame(test_predict, columns=['feature_2']).to_csv('myprediction.csv',index = None)