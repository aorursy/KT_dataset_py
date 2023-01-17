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
import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
data = pd.read_csv('../input/climate-change/climate_change.csv')
data
data.info()
data.isna().sum()
plt.scatter(data['CO2'], data['Temp'])
plt.bar(data['Year'], data['Temp'])
plt.bar(data['Year'],data['Aerosols'])
plt.bar(data['Year'],data['CO2'])
data.describe()
data.corr()
data.drop('Month', axis=1,inplace=True)
def model(m):
    np.random.seed(0)
    x = data.drop('CO2', axis=1)
    y = data['CO2']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    clf = m
    clf.fit(x_train, y_train)
    s = clf.score(x_test,y_test)
    return s
model(RandomForestRegressor())
model(LinearRegression())
np.random.seed(45)
x = data.drop('CO2', axis=1)
y = data['CO2']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
clf = LinearRegression()
clf.fit(x_train, y_train)
clf.score(x_test,y_test)
y_preds = clf.predict(x_test)
mean_absolute_error(y_test, y_preds)
grid={'n_estimators': [10,100,200,500,1000,1200], 
      'max_depth':[None, 5, 10, 20, 30], 
      'max_features':[0.5, 0.2, 'auto', 'sqrt'],
      'min_samples_split':[2,4,6],
      'min_samples_leaf':[1,2,4]}
np.random.seed(45)
x = data.drop('CO2', axis=1)
y = data['CO2']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
clf = RandomForestRegressor()
rs_clf = RandomizedSearchCV(estimator=clf,
                            param_distributions=grid,
                            n_iter=50, 
                            cv=5,
                            verbose=2)
rs_clf.fit(x_train, y_train)
rs_clf.score(x_test,y_test)
y_preds2 = rs_clf.predict(x_test)
mean_absolute_error(y_test, y_preds2)
