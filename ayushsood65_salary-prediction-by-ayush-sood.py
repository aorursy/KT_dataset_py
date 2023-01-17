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
data=pd.read_csv("../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv",index_col=0)
data
data['specialisation'].unique()
data
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
data['gender']=encoder.fit_transform(data['gender'])
data
encoder1=LabelEncoder()
data['ssc_b']=encoder1.fit_transform(data['ssc_b'])
data['hsc_b']=encoder1.fit_transform(data['hsc_b'])
data
data['workex']=encoder1.fit_transform(data['workex'])
data['specialisation']=encoder1.fit_transform(data['specialisation'])
data['status']=encoder1.fit_transform(data['status'])
data
data=pd.get_dummies(data,columns=['hsc_s','degree_t'])

data
X=data.drop(columns=['salary'])
Y=data['salary'].fillna(0)
X.isnull().sum()
Y.isnull().sum()
X.drop(columns=['hsc_s_Arts','degree_t_Comm&Mgmt'])
Y.head()
data.corr()
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
from xgboost import XGBRegressor
regressor=XGBRegressor(n_estimators=100000)
regressor.fit(X_train,Y_train)
y_pred=regressor.predict(X_test)
from sklearn.metrics import r2_score
accuracy=r2_score(y_pred,Y_test)
accuracy
