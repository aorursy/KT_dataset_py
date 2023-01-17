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
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_absolute_error

import warnings

warnings.filterwarnings("ignore")
df_iris = pd.read_csv('../input/iris/Iris.csv')
df_iris.describe()
df_iris.head()
df_iris.columns
df_iris.shape
df_iris.isnull().sum()
df_iris.select_dtypes('object').head()
y = df_iris.Species
y.head()
y = pd.get_dummies(y)
y.head()
X = df_iris.drop(columns = ['Species'],axis = 0)
X.head()
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0)
X_train.head()
model_RF = RandomForestRegressor(random_state=1)
model_RF.fit(X_train, y_train)
predictions = model_RF.predict(X_val)
predictions
mae = mean_absolute_error(y_val, predictions)
print(" MAE score : ", mae)