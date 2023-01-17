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
df = pd.read_csv('/kaggle/input/automobile-dataset/Automobile_data.csv')
df.head()
df.columns
df['num-of-cylinders'] = encoder.fit_transform(df['num-of-cylinders'])
df = df.drop(['engine-location'],axis = 1)
df['fuel-system'] = encoder.fit_transform(df['fuel-system'])
df['engine-type'] = encoder.fit_transform(df['engine-type'])
import numpy as np

df = df.replace('NaN','np.nan')
df.shape
from sklearn.preprocessing import StandardScaler,LabelEncoder

encoder = LabelEncoder()

scaler = StandardScaler()
df['fuel-system'].value_counts()
x = df.drop(['price'],axis = 1)
x = scaler.fit_transform(x)
x = np.asarray(x)
y = np.asarray(df['price'])

df['make'] = encoder.fit_transform(df['make'])
df['fuel-type'] = encoder.fit_transform(df['fuel-type'])
df['aspiration'] = encoder.fit_transform(df['aspiration'])
df['num-of-doors'] = encoder.fit_transform(df['num-of-doors'])
df['body-style'] = encoder.fit_transform(df['body-style'])
df['drive-wheels'] = encoder.fit_transform(df['drive-wheels'])
from sklearn.model_selection import train_test_split

x_train,y_train,x_test,y_test = train_test_split(x,y,test_size = 0.1)
!pip install xgboost
from xgboost import XGBRegressor

from sklearn.linear_model import LinearRegression

model = XGBRegressor()

model.fit(x_train,x_test)
model.score(y_train,y_test)
y_pred = model.predict(y_train)
y_pred[0]
y_test[0]