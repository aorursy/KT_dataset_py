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
!ls /kaggle/input/random-linear-regression/train.csv
import pandas as pd
train_data = pd.read_csv('/kaggle/input/random-linear-regression/train.csv')
test_data = pd.read_csv('/kaggle/input/random-linear-regression/test.csv')
train_data.head()
train_data.isnull().sum()
train_data['y'] = train_data['y'].fillna(train_data['y'].mean())
train_data.isnull().sum()
x = pd.DataFrame(train_data['x'])
y = pd.DataFrame(train_data['y'])
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(x,y)
reg.intercept_
reg.coef_
val_pred = reg.predict(y)
val_pred[0:5]
import matplotlib.pyplot as plt
%matplotlib inline

plt.scatter(x,y)
plt.ylim([0,80])
plt.xlim([0,100])
plt.plot(y, val_pred)
plt.show()
from sklearn import metrics

print("MAE :- ", metrics.mean_absolute_error(y,val_pred))
print("MSE :- ", metrics.mean_squared_error(y,val_pred))
print("RMSE :- ", np.sqrt(metrics.mean_squared_error(y,val_pred)))
reg.score(x,y)
