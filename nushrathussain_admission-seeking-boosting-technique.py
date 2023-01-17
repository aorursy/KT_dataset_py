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
import seaborn as sns
data=pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict.csv')
data.head()
data.isna().sum()
data.drop(['Serial No.'],axis=1,inplace=True)
sns.lineplot(data['GRE Score'],data['Chance of Admit '])
sns.countplot(data['Research'])
data['University Rating']=data['University Rating'].astype(str)

rating=pd.get_dummies(data['University Rating'],prefix='rating_',drop_first=True)

data.drop(['University Rating'],axis=1,inplace=True)
new_data=data.iloc[:,:2].copy()

new_data=pd.concat([new_data,rating],axis=1)

new_data=pd.concat([new_data,data.iloc[:,2:].copy()],axis=1)
new_data.head()
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import GradientBoostingRegressor
train_label=new_data['Chance of Admit '][0:300]

train_features=new_data.drop('Chance of Admit ',axis=1)[0:300]



test_label=new_data['Chance of Admit '][300:]

test_features=new_data.drop('Chance of Admit ',axis=1)[300:]
gbr=GradientBoostingRegressor(loss='huber',learning_rate=0.1,max_depth=3,alpha=0.82,random_state=0,n_estimators=70)

gbr.fit(train_features,train_label)

y_pred=gbr.predict(test_features)

print('MSE : ',mean_squared_error(test_label,y_pred))

print('RMSE : ',np.sqrt(mean_squared_error(test_label,y_pred)))