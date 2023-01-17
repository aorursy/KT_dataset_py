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
train=pd.read_csv('/kaggle/input/ihs-markit-sample-competition/train_sample.csv')#.set_index('vehicle_id')
print(data.head())
test=pd.read_csv('/kaggle/input/ihs-markit-sample-competition/train_sample.csv')
print(test.head())
train.corr()
train.describe()
train.plot.scatter(y='Curb_Weight',x='Price_USD', figsize=(10,5))
#modelling on train data first

X=train[['Curb_Weight','year']]
#X=data['Curb_Weight'].values.reshape(-1,1)
y=train['Price_USD'].values
y.shape
X=np.array(X)
y=np.array(y)
def mape(y_true, y_pred):
    y_val = np.maximum(np.array(y_true), 1e-8)
    return (np.abs(y_true -y_pred)/y_val).mean()
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test=train_test_split(X, y, random_state=0, test_size=.3)
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
X_train_scaled=StandardScaler().fit_transform(X_train)
X_test_scaled=StandardScaler().fit_transform(X_test)
decision=RandomForestRegressor(random_state=0,  max_depth=None, min_samples_split=2)
decision.fit(X_train_scaled, y_train)
y_predict=decision.predict(X_test)
print('Mean Absolute Percentage Error for scaled RandomForest Model: ',mape(y_test, y_predict))
y_predict
test_X=test[['Curb_Weight','year']]
test_X_scaled=StandardScaler().fit_transform(test_X)
test_predict=decision.fit(test_X_scaled,y_train)


