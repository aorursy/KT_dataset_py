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
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import OneHotEncoder
data=pd.read_csv('../input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv')
data.info()
data.describe()
data.isnull().any()
data1=data.drop(['PassengerId','Firstname','Lastname'],axis=1)
data1.info()
col=['Country','Sex','Age','Category']
X_train,X_valid,y_train,y_valid=train_test_split(data1[col],data1['Survived'],train_size=0.8,test_size=0.2,random_state=0)
hot_col=['Country','Sex','Category']

OH_encoder=OneHotEncoder(handle_unknown='ignore',sparse=False)

OH_train=pd.DataFrame(OH_encoder.fit_transform(X_train[hot_col]))

OH_valid=pd.DataFrame(OH_encoder.transform(X_valid[hot_col]))

OH_train.index=X_train.index

OH_valid.index=X_valid.index
OH_train['Age']=X_train['Age']

OH_train.info()
OH_valid['Age']=X_valid['Age']

OH_valid.info()
forest_model=RandomForestRegressor(n_estimators=10000,random_state=1)

forest_model.fit(OH_train,y_train)

prediction=forest_model.predict(OH_valid)

print(mean_absolute_error(y_valid,prediction))