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
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
file_path = '../input/concrete-compressive-strength/Concrete_Data.xls'
data = pd.read_excel(file_path)
data.head()
features = data.iloc[:,:-1]
features.head()
plt.figure(figsize=(10,5))
sns.heatmap(features.corr(), cmap ='Blues', annot = True)
features.isnull().sum()
data.info()
data.describe()
X = features.copy().values
y = data.iloc[:,-1].values

train_X, test_X, train_y, test_y = train_test_split(X,y, random_state = 1)
from sklearn import linear_model
lr_model = linear_model.LinearRegression()
lr_model.fit(train_X, train_y)
predict_y = lr_model.predict(test_X)
from sklearn.svm import SVR
svr_model = SVR(C=0.8, kernel = 'linear')
svr_model.fit(train_X, train_y)
predict_y = svr_model.predict(test_X)
from sklearn.linear_model import BayesianRidge
br_model = BayesianRidge(compute_score = True)
br_model.fit(train_X, train_y)
predict_y = br_model.predict(test_X)
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(random_state = 1)
rf_model.fit(train_X,train_y)
predict_y = rf_model.predict(test_X)
from sklearn.tree import DecisionTreeRegressor
dt_model = DecisionTreeRegressor(random_state = 1)
dt_model.fit(train_X,train_y)
predict_y = dt_model.predict(test_X)
models = [lr_model, svr_model, br_model, rf_model, dt_model]
model_names = ['Linear Regression Model', 'Support Vector Regression Model', 'BayesianRidge Regression Model', 'Random Forest Regression Model', 'Decision Tree Regression Model']
for model,model_name in zip(models,model_names):
    mae = mean_absolute_error(model.predict(test_X),test_y)
    print(f"{model_name} MEA = {mae}")
for model,model_name in zip(models,model_names):
    accuracy = model.score(test_X,test_y)
    print(f"{model_name} accuracy score = {accuracy}")
for model,model_name in zip(models,model_names):
    plt.plot(test_y, label = 'Actual Target')
    plt.plot(model.predict(test_X), label = 'Predicted Target')
    plt.legend(loc = 'upper left')
    plt.title(model_name)
    plt.show()