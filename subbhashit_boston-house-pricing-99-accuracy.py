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
data=pd.read_csv("/kaggle/input/boston-house-prices/housing.csv",names=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'], header=None, delimiter=r"\s+")
data.head()
data.info()
data.describe()
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use("classic")
sns.heatmap(data.corr(),annot=True,cmap='inferno')
sns.pairplot(data)
x=data.drop("MEDV",axis=1)
y=data["MEDV"]
from sklearn.model_selection import train_test_split,KFold,cross_val_score
xr,xt,yr,yt=train_test_split(x,y)
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier,RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBClassifier,XGBRFRegressor,XGBRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression,LinearRegression,SGDRegressor
from sklearn.svm import SVC,SVR
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMRegressor
model=XGBRegressor(n_estimators=1000)
model.fit(x,y)
kfold=KFold(n_splits=10)
res=cross_val_score(model,x,y,cv=kfold)
res.mean()*100
yp=model.predict(xt)
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print(r2_score(yt,yp))
print(mean_absolute_error(yt,yp))
print(mean_squared_error(yt,yp))