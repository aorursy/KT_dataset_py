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
df = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')
df.head()
X = df.loc[:,'GRE Score':'Research'].values
y = df.iloc[:,-1].values
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

lr = LinearRegression()
sgdr = SGDRegressor()
mlpreg = MLPRegressor(random_state=1, max_iter=100)
svrl = SVR(kernel='linear',C=0.01)
svrp = SVR(kernel='poly',degree=3,C=1)
models = {'LinReg':lr,'SGDReg':sgdr,'MLPreg':mlpreg,'SVRLinear':svrl,'SVRPoly':svrp}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, shuffle=None)
scaler = StandardScaler()  #MinMaxScaler(feature_range=(0,1))

for name in models:
    regressor = models[name]
    pipeline = Pipeline(steps=[('scaler',scaler),('name',regressor)])
    model = pipeline.fit(X_train,y_train)
    score = model.score(X_test,y_test)
    predict = model.predict(X_test)
    rmse = mean_squared_error(y_test,predict, squared=False)
    
    print(name+': score - %1.3f, rmse - %1.4f'%(score,rmse))

import seaborn as sns
import matplotlib.pyplot as plt

df1 = df.loc[:,'GRE Score':'Chance of Admit ']
grid = sns.PairGrid(df1)
grid.map(plt.scatter)
X = df[['TOEFL Score','SOP','CGPA','Research']].values
y = df.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, shuffle=None)
for name in models:
    regressor = models[name]
    pipeline = Pipeline(steps=[('scaler',scaler),('name',regressor)])
    model = pipeline.fit(X_train,y_train)
    score = model.score(X_test,y_test)
    predict = model.predict(X_test)
    rmse = mean_squared_error(y_test,predict, squared=False)
    
    print(name+': score - %1.3f, rmse - %1.4f'%(score,rmse))