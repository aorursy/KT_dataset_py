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
demo=pd.read_csv("/kaggle/input/california-housing-prices/housing.csv")
demo.head()
from matplotlib import pyplot as plt
import seaborn as sns
sns.distplot(demo['households'])
demo.describe()
null_count=demo.isnull().sum()
print(null_count)
demo['total_bedrooms']
demo['ocean_proximity'].unique()

demo = demo.drop('ocean_proximity',axis=1)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
it = IterativeImputer(estimator = RandomForestRegressor())
newdemo = pd.DataFrame(it.fit_transform(demo))
newdemo.columns = demo.columns
newdemo.head()
# it = IterativeImputer(estimator = LinearRegression())
# newdemo = pd.DataFrame(it.fit_transform(demo))
#newdemo.columns = demo.columns
# newdemo.head()
null_count=newdemo.isnull().sum()
print(null_count)
newdemo.describe()
sns.distplot(newdemo['total_bedrooms'])
sns.distplot(demo['total_bedrooms'])
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer()
transformed_data=pt.fit_transform(newdemo[['total_bedrooms']])
sns.distplot(transformed_data)
#Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
demo = pd.read_csv("/kaggle/input/california-housing-prices/housing.csv")
demo = demo.drop('ocean_proximity',axis=1)
X = demo.drop('median_house_value',axis=1)
y = demo[['median_house_value']]
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size= .20,random_state=10)
pipe = Pipeline((
("it", IterativeImputer(estimator = LinearRegression())),
("pt", PowerTransformer()),
("sc",StandardScaler()),

("lr", XGBRegressor(n_estimators=100)),
))
pipe.fit(Xtrain,ytrain)
print("Training R2")
print(pipe.score(Xtrain,ytrain))
print("Testing R2")
print(pipe.score(Xtest,ytest))
from sklearn.model_selection import cross_val_score
scoresxgb = cross_val_score(pipe,Xtrain,ytrain,cv=10,scoring='r2')
print(scoresxgb)

print("Average:")
print(np.mean(scoresxgb))
scoresxgbtest = cross_val_score(pipe,Xtest,ytest,cv=10,scoring='r2')
print(scoresxgbtest)
print("Average:")
print(np.mean(scoresxgbtest))