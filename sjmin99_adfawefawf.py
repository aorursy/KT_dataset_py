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
from datetime import datetime

def year_to_month(pd):
    
    year = np.array(pd['year'],dtype=str)

    dateFormatter = "%Y%m%d"

    month = list()
    for p in year:
        month.append(datetime.strptime(p, dateFormatter).strftime('%m'))
    
    
    print(len(year), len(month))
    
    return month
pd_train = pd.read_csv("/kaggle/input/mlregression-cabbage-price/train_cabbage_price.csv")
pd_test = pd.read_csv("/kaggle/input/mlregression-cabbage-price/test_cabbage_price.csv")

train_month = year_to_month(pd_train)
test_month = year_to_month(pd_test)

pd_train['year'] = train_month 
pd_test['year'] = test_month

pd_train.rename(columns = {"year":"month"}, inplace = True)
pd_test.rename(columns = {"year":"month"}, inplace = True)

train_x = np.array(pd_train.iloc[:,:-1], dtype=float)
train_y = np.array(pd_train.iloc[:,-1]).reshape(-1,1)

test_data = np.array(pd_test, dtype = float)

print(train_x.shape, train_y.shape, test_data.shape)
from sklearn.preprocessing import normalize

train_x[:,1:], l2norm = normalize(train_x[:,1:], axis = 0, return_norm = True)
test_data[:,1:] = test_data[:,1:] / l2norm

# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# sc.fit(train_x[:,1:])
# train_x[:,1:] = sc.transform(train_x[:,1:])
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split


X_train, X_valid, Y_train, Y_valid = train_test_split(train_x, train_y, test_size = 0.2, random_state = 1)
print(X_train.shape, X_valid.shape,'\n',Y_train.shape, Y_valid.shape)

knn = KNeighborsRegressor(n_neighbors=15, p=2)
# knn.fit(X_train, Y_train)
# valid_predict = knn.predict(X_valid)

# from sklearn.metrics import mean_squared_error

# print(mean_squared_error(Y_valid, valid_predict) ** 0.5)

knn.fit(train_x, train_y)
test_predict = knn.predict(test_data)

Id = np.array([i for i in range(len(test_predict))]).reshape(-1,1)
Expected = test_predict.reshape(-1,1)

result = np.hstack((Id, Expected))

result.shape
df = pd.DataFrame(result, columns={"Id","Expected"}, dtype = "int32")
# df['Id'] = df['Id'].astype('int32')
df.to_csv("result.csv", index = False, header = True)
