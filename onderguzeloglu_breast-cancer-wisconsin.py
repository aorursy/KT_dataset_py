# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")
df.columns
df.info()
# create data 1 that includes perimeter_worst target variable area_worst
data1 = df[df['diagnosis'] == 'M']

x = np.array(data1.loc[:,'perimeter_worst']).reshape(-1,1)
y = np.array(data1.loc[:,'area_worst']).reshape(-1,1)

sns.jointplot(x = "perimeter_worst", y = "area_worst", data =df)
# Linear regression 
from sklearn.linear_model import LinearRegression
reg = LinearRegression()

predict_space = np.linspace(min(x),max(x)).reshape(-1,1)
reg.fit(x,y)
predicted = reg.predict(predict_space)
print('R^2 score: ', reg.score(x,y))
print(reg.predict([[165]]))
sns.jointplot(x="perimeter_worst", y = "area_worst", data = df, kind= "reg")
# sklearn library
from sklearn.linear_model import LinearRegression

x = df.iloc[:,[24,25]].values
y = df.texture_worst.values.reshape(-1,1)

multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(x,y)

print("b0 : ", multiple_linear_regression.intercept_)
print("b1,b2 : ", multiple_linear_regression.coef_)

multiple_linear_regression.predict(np.array([[160,1850],[90,1850]]))
from sklearn.metrics import r2_score
y = df.area_worst.values.reshape(-1,1)
x = df.perimeter_worst.values.reshape(-1,1)

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state = 42)

rf.fit(x,y)

y_head = rf.predict(x)
print(rf.predict([[165]]))
from sklearn.metrics import r2_score

print("r_score: ", r2_score(y,y_head))
df