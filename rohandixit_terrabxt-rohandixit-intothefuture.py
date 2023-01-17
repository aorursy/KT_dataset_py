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
import matplotlib.pyplot as plt
# Load the datasets
df_train = pd.read_csv("/kaggle/input/into-the-future/train.csv")
df_test = pd.read_csv("/kaggle/input/into-the-future/test.csv")
df_train.head()
df_train.drop(["id"], axis = 1, inplace = True)
# Convert time column to datetime datatype
df_train['time'] = pd.to_datetime(df_train['time'])
#set index
df_train.set_index('time', inplace = True)
df_train.head()
#assign X and Y
X = df_train[['feature_1']]
y = df_train[['feature_2']]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)
from sklearn.linear_model import LinearRegression
Linear_regression = LinearRegression()
Linear_regression.fit(X_train,y_train)
df_test
# Convert time column to datetime datatype
df_test['time'] = pd.to_datetime(df_train['time'])
#set index
df_test.set_index('time', inplace = True)
df_test.head()
X_test1=df_test[['feature_1']]
Prediction=Linear_regression.predict(X_test1)
Prediction
print(len(Prediction))
print(Prediction.shape)
df_result=pd.DataFrame({"id": df_test['id'], "feature_2_Prediction":Prediction.ravel()})
df_result.to_csv("/kaggle/working/prediction.csv",index=False)
