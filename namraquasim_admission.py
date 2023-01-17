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
import pandas as pd
import numpy as nm
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv("../input/graduate-admissions/Admission_Predict.csv")
data
verify = pd.read_csv("../input/graduate-admissions/Admission_Predict_Ver1.1.csv")
verify
data.count()
from sklearn.model_selection import train_test_split

y=data[data.columns[8]].values;
X=data[data.columns[[1,2,3,4,5,6,7]]].values

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train.shape)
print(X_test.shape)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

model=RandomForestRegressor()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)
print("Accuracy: ", model.score(X_test,y_test))

print("MSE: ", mean_squared_error(y_pred,y_test))
print("RMSE: ", nm.sqrt(mean_squared_error(y_pred,y_test)))

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)
print("Accuracy: ", model.score(X_test,y_test))


print("MSE: ", mean_squared_error(y_pred,y_test))
print("RMSE: ", nm.sqrt(mean_squared_error(y_pred,y_test)))
from sklearn.model_selection import train_test_split

y1=verify[verify.columns[8]].values;
X1=verify[verify.columns[[1,2,3,4,5,6,7]]].values

print(X1.shape)
print(y1.shape)

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2)

print(X1_train.shape)
print(X1_test.shape)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

model=RandomForestRegressor()
model.fit(X1_train,y1_train)

y1_pred=model.predict(X1_test)
print("Accuracy: ", model.score(X1_test,y1_test))

print("MSE: ", mean_squared_error(y1_pred,y1_test))
print("RMSE: ", nm.sqrt(mean_squared_error(y1_pred,y1_test)))

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X1_train,y1_train)

y1_pred=model.predict(X1_test)
print("Accuracy: ", model.score(X1_test,y1_test))


print("MSE: ", mean_squared_error(y1_pred,y1_test))
print("RMSE: ", nm.sqrt(mean_squared_error(y1_pred,y1_test)))
